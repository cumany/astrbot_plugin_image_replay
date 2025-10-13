from __future__ import annotations

import asyncio
import base64
import imghdr
import json
import random
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType, event_message_type
from astrbot.api.message_components import Image, Node
from astrbot.api.star import Context, Star, register,StarTools
from astrbot.core.message.components import Reply



SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


@register(
    "image_replay",
    "cuman",
    "支持自定义表情，图片，随机，批量发送。解决表情制作困难的问题。",
    "1.0.0",
    "https://github.com/cuman/astrbot_plugin_image_replay",
)
class ImageReplayPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        DATA_DIR = StarTools.get_data_dir('astrbot_plugin_image_replay')
        self._resource_dir = DATA_DIR / 'resource'
        self.config = config
        self._ensure_resource_dir()

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def handle_message(self, event: AstrMessageEvent):
        try:
            if not self._passes_message_guards(event, log_on_skip=True):
                return

            message = (event.message_str or "").strip()
            if not message:
                return

            logger.debug(f"Received message: {message}")
            raw_commands = self.config.get("commands", [])
            # 精确匹配：检查关键字是否等于消息字符串的第一个单词
            keyword = next(
                (k for k in raw_commands if k == message.split()[0]), None
            )

            if not keyword:
                return

            images = self._collect_group_images(keyword)
            if not images:
                yield event.plain_result(
                    f"未找到与前缀“{keyword}”匹配的图片，请检查 resource 目录。"
                )
                return
            response = self._prepare_image_response(event, images, keyword)
            if response is None:
                return
            kind, payload = response
            if kind == "single":
                yield event.chain_result([Image.fromFileSystem(str(payload))])
            else:
                yield event.chain_result(payload)
            return
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    # 已移除基于消息文本的 "%搜集" 处理器，保留基于命令的 @filter.command("收集") 处理器

    @filter.command("查看图片")
    async def handle_view_command(self, event: AstrMessageEvent):
        if not self._is_allowed(event):
            return
        images = self._collect_all_images()
        if not images:
            yield event.plain_result("未找到可用图片，请先在 resource 目录添加资源。")
            return
        response = self._prepare_image_response(event, images, "图片内容")
        if response is None:
            yield event.plain_result("未找到可用图片，请先在 resource 目录添加资源。")
            return
        kind, payload = response
        if kind == "single":
            yield event.chain_result([Image.fromFileSystem(str(payload))])
        else:
            yield event.chain_result(payload)

    @filter.command("收集",alias={'添加图片', '添加表情'})
    async def handle_collect_command_cmd(self, event: AstrMessageEvent):
        """使用命令触发的收集：命令后的文本作为前缀（文件名开头）。

        例：
        收集 可爱表情
        将把引用消息中的图片保存为前缀为 `可爱表情` 的文件。
        """
        if not self._is_allowed(event):
           return
        try:
            message = (event.message_str or "").strip()
            _, args = self._split_command_args(message)
            prefix = self._normalize_collect_prefix(args or "临时")

            # 将前缀加入指令列表并保存到配置（首选通过框架配置）
            current = self.config.get("commands",[])
            if prefix and prefix not in current:
                current.append(prefix)
            self.config["commands"] = current  # type: ignore[index]
            save_fn = getattr(self.config, "save_config", None)
            if callable(save_fn):
                save_fn()       
            images = await self._load_images_from_event(event)
            if not images:
                yield event.plain_result("未在引用消息中找到图片，请先引用需要收集的图片。")
                return

            saved_files: List[str] = []
            for data, extension in images:
                target_path = self._next_available_file(prefix, extension)
                target_path.write_bytes(data)
                saved_files.append(target_path.name)

            file_list = "、".join(saved_files)
            yield event.plain_result(f"已收集 {len(saved_files)} 张图片，文件名：{file_list}")
        except Exception as exc:
            logger.error(f"收集图片失败(命令): {exc}", exc_info=True)
            yield event.plain_result("收集图片时出现异常，请稍后重试。")

   
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("删除图片指令",alias={'删除图片'})
    async def handle_delete_keyword(self, event: AstrMessageEvent):
        """命令格式：删除图片指令 <关键词> [ALL]

        - 不带 ALL：删除与关键词前缀匹配的最近添加的一张图片（按文件名的时间戳/排序逆序判断最近）
        - 带 ALL：删除所有与关键词前缀匹配的图片
        """
        try:
            text = (event.message_str or "").strip()
            if not text:
                yield event.plain_result("请提供关键词，例如：删除图片指令 可爱表情 或 删除图片指令 可爱表情 ALL")
                return
            _, args = self._split_command_args(text)
            parts = args.split()
            if not parts:
                yield event.plain_result("请提供关键词，例如：删除图片指令 可爱表情")
                return
            keyword = parts[0].strip()
            flag_all = len(parts) > 1 and parts[1].upper() == "ALL"

            # 寻找资源目录下匹配前缀的文件（文件名以 prefix-... 的形式）
            matches: List[Path] = []
            for path in sorted(self._resource_dir.iterdir(), key=lambda p: p.name):
                if not path.is_file():
                    continue
                stem = path.stem
                if '-' not in stem:
                    continue
                prefix = stem.split('-', 1)[0]
                if prefix == keyword:
                    matches.append(path)

            if not matches:
                yield event.plain_result(f"未找到与前缀'{keyword}'匹配的图片。")
                return

            if flag_all:
                removed = 0
                for p in matches:
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        logger.warning(f"删除文件失败: {p}")
                # 如果删除后该前缀没有剩余图片，从 commands 中移除前缀
                self._maybe_remove_command_prefix(keyword)
                yield event.plain_result(f"已删除 {removed} 张与 '{keyword}' 匹配的图片（ALL 模式）。")
                return

            # 非 ALL：删除最近添加的（按文件名或者文件修改时间决定）
            # 使用修改时间作为最近性判定
            matches_sorted = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
            target = matches_sorted[0]
            try:
                target.unlink()
                # 删除后如无剩余图片，则从 commands 中移除前缀
                self._maybe_remove_command_prefix(keyword)
                yield event.plain_result(f"已删除最近添加的图片：{target.name}")
            except Exception as exc:
                logger.error(f"删除文件失败: {exc}", exc_info=True)
                yield event.plain_result("删除图片失败，请检查文件权限或稍后再试。")
        except Exception as exc:
            logger.error(f"删除图片指令失败: {exc}", exc_info=True)
            yield event.plain_result("删除图片指令时出现异常，请稍后重试。")

    def _ensure_resource_dir(self) -> None:
        self._resource_dir.mkdir(parents=True, exist_ok=True)


    def _collect_all_images(self) -> List[Path]:
        if not self._resource_dir.exists():
            return []
        results: List[Path] = []
        for path in sorted(self._resource_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            results.append(path)
        return results

    def _select_images(self, images: List[Path]) -> List[Path]:
        if not images:
            return []
        multi_mode = self.config.get("multi_reply_mode", "random")
        if multi_mode == "all" or len(images) == 1:
            return images
        # 使用 random.sample 明确从列表中随机取一项
        try:
            pick = random.sample(images, 1)
            return pick
        except Exception:
            return [random.choice(images)]

    def _collect_group_images(self, group: str) -> List[Path]:
        results: List[Path] = []
        if not self._resource_dir.exists():
            return results

        for path in sorted(self._resource_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            stem = path.stem
            if "-" not in stem:
                continue
            prefix = stem.split("-", 1)[0]
            if prefix != group:
                continue
            results.append(path)
        return results

    async def _load_images_from_event(
        self, event: AstrMessageEvent
    ) -> List[Tuple[bytes, str]]:
        payloads: List[Tuple[bytes, str]] = []
        for segment in self._gather_image_segments(event):
            loaded = await self._load_image_segment(segment)
            if loaded:
                payloads.append(loaded)
        return payloads

    def _gather_image_segments(self, event: AstrMessageEvent) -> List[Image]:
        chain = event.get_messages()
        reply_seg = next((seg for seg in chain if isinstance(seg, Reply)), None)
        if reply_seg and getattr(reply_seg, "chain", None):
            reply_images = [seg for seg in reply_seg.chain if isinstance(seg, Image)]
            if reply_images:
                return reply_images
        return [seg for seg in chain if isinstance(seg, Image)]

    async def _load_image_segment(
        self, segment: Image
    ) -> Optional[Tuple[bytes, str]]:
        candidates = [getattr(segment, "url", None), getattr(segment, "file", None)]
        for src in candidates:
            if not src:
                continue
            data = await self._load_image_source(src)
            if data:
                extension = await self._derive_image_extension(data, src)
                return data, extension
        return None

    async def _load_image_source(self, src: str) -> Optional[bytes]:
        if src.startswith("base64://"):
            try:
                return base64.b64decode(src[9:])
            except Exception as exc:
                logger.error(f"解码 Base64 图片失败: {exc}")
                return None
        parsed = urlparse(src)
        try:
            if parsed.scheme in {"http", "https"}:
                return await self._download_image(src)
            if parsed.scheme == "file":
                file_path = Path(parsed.path)
                return file_path.read_bytes() if file_path.is_file() else None
            path = Path(src)
            return path.read_bytes() if path.is_file() else None
        except Exception as exc:
            logger.error(f"读取图片资源失败: {exc}")
            return None

    async def _download_image(self, url: str) -> Optional[bytes]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.read()
        except Exception as exc:
            logger.error(f"图片下载失败: {exc}")
            return None

    async def _derive_image_extension(self, data: bytes, src: Optional[str]) -> str:
        suffix = self._suffix_from_source(src) if src else None
        if suffix:
            return suffix
        detected = await asyncio.to_thread(imghdr.what, None, data)
        if detected:
            if detected == "jpeg":
                detected = "jpg"
            candidate = f".{detected}"
            if candidate in SUPPORTED_EXTENSIONS:
                return candidate
        return ".jpg"

    def _suffix_from_source(self, src: str) -> Optional[str]:
        if not src or src.startswith("base64://"):
            return None
        parsed = urlparse(src)
        suffix = ""
        if parsed.scheme in {"http", "https", "file"}:
            suffix = Path(parsed.path).suffix
        else:
            suffix = Path(src).suffix
        suffix = suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            return suffix
        return None

    def _normalize_collect_prefix(self, prefix: str) -> str:
        candidate = prefix.strip() or "临时"
        invalid_chars = '<>:"/\\|?*'
        sanitized = "".join(
            "_" if ch in invalid_chars else ch for ch in candidate
        ).replace(" ", "_")
        sanitized = sanitized.strip("-") or "临时"
        return sanitized[:32]

    def _next_available_file(self, prefix: str, extension: str) -> Path:
        safe_extension = extension if extension in SUPPORTED_EXTENSIONS else ".jpg"
        for _ in range(50):
            suffix = f"{random.randint(0, 9999):04d}"
            candidate = self._resource_dir / f"{prefix}-{suffix}{safe_extension}"
            if not candidate.exists():
                return candidate
        suffix = f"{random.randint(0, 999999):06d}"
        return self._resource_dir / f"{prefix}-{suffix}{safe_extension}"

    def _maybe_remove_command_prefix(self, prefix: str) -> None:
        """如果 resource 目录下不再存在以 prefix 开头的图片，则从配置/本地文件中移除该 prefix。"""
        # 检查是否还有匹配文件
        has_any = False
        for path in self._resource_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            if '-' not in stem:
                continue
            pfx = stem.split('-', 1)[0]
            if pfx == prefix:
                has_any = True
                break
        if has_any:
            return
        current = self.config.get("commands",[])
        if prefix in current:
            current = [c for c in current if c != prefix]
            self.config["commands"] = current  # type: ignore[index]
            save_fn = getattr(self.config, "save_config", None)
            if callable(save_fn):
                save_fn()
            return


    def _prepare_image_response(
        self,
        event: AstrMessageEvent,
        images: List[Path],
        label: str,
    ) -> Optional[Tuple[str, object]]:
        selected_images = self._select_images(images)
        if not selected_images:
            return None
        if len(selected_images) == 1:
            return ("single", selected_images[0])
        forward_nodes = self._build_forward_nodes(event, selected_images, label)
        if not forward_nodes:
            return None
        return ("forward", forward_nodes)

    def _build_forward_nodes(
        self, event: AstrMessageEvent, images: Iterable[Path], command: str
    ) -> List[Node]:
        uin, default_name = self._resolve_bot_identity(event)
        node_name = command if command else default_name
        nodes: List[Node] = []
        for image in images:
            nodes.append(
                Node(
                    uin=uin,
                    name=node_name,
                    content=[Image.fromFileSystem(str(image))],
                )
            )
        return nodes

    def _resolve_bot_identity(self, event: AstrMessageEvent) -> Tuple[str, str]:
        message_obj = getattr(event, "message_obj", None)
        self_id = getattr(message_obj, "self_id", None) if message_obj else None
        if self_id is None:
            self_id = event.get_sender_id()
        return str(self_id), "ImageReplay"

    def _passes_message_guards(
        self, event: AstrMessageEvent, *, log_on_skip: bool = False
    ) -> bool:
        require_wake =self.config.get("require_wake", True)
        
        if require_wake and not getattr(event, "is_at_or_wake_command", False):
            if log_on_skip:
                logger.debug("Not a wake command, skipping.")
            return False
        if not self._is_allowed(event):
            return False
        return True

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        group_id = event.get_group_id() or ""
        user_id = str(event.get_sender_id() or "")
        group_whitelist = self.config.get("group_whitelist",[])
        group_blacklist = self.config.get("group_blacklist",[])
        user_whitelist = self.config.get("user_whitelist",[])
        user_blacklist = self.config.get("user_blacklist",[])
        if group_whitelist and group_id not in group_whitelist:
            return False
        if group_blacklist and group_id and group_id in group_blacklist:
            return False

        if user_whitelist and user_id not in user_whitelist:
            return False
        if user_blacklist and user_id in user_blacklist:
            return False

        return True



    def _split_command_args(self, message: str) -> Tuple[str, str]:
        trimmed = (message or "").strip()
        if not trimmed:
            return "", ""
        parts = trimmed.split(maxsplit=1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]
