import unittest
import re
from pathlib import Path


class UIPreviewTestCase(unittest.TestCase):
    def setUp(self):
        self.html = Path("ui_preview.html").read_text(encoding="utf-8")

    def test_preview_uses_chat_first_layout(self):
        self.assertIn('class="chat-app"', self.html)
        self.assertIn('class="chat-main"', self.html)
        self.assertIn("医疗问答助手", self.html)

    def test_default_mode_is_local_retrieval(self):
        self.assertIn("本地检索模式", self.html)
        self.assertIn('data-mode="local"', self.html)

    def test_retrieval_evidence_is_collapsed_by_default(self):
        self.assertIn("<details", self.html)
        self.assertIn("查看检索依据", self.html)
        self.assertNotIn('class="main-grid"', self.html)

    def test_voice_input_uses_microphone_svg_icon(self):
        self.assertIn('aria-label="语音输入"', self.html)
        self.assertIn('class="mic-icon"', self.html)
        self.assertIn("<svg", self.html)
        self.assertNotIn(">麦</button>", self.html)

    def test_layout_has_responsive_viewport_optimizations(self):
        self.assertIn("--rail-width", self.html)
        self.assertIn("--composer-height", self.html)
        self.assertIn("padding-bottom: calc(var(--composer-height)", self.html)
        self.assertIn("@media (max-width: 720px)", self.html)

    def test_header_status_controls_are_visually_subtle(self):
        self.assertIn('class="header-pill status-chip"', self.html)
        self.assertIn('class="status-dot"', self.html)
        self.assertIn('class="header-pill plain-button"', self.html)
        status_rule = re.search(r"\.status-chip\s*\{(?P<body>[^}]+)\}", self.html)
        self.assertIsNotNone(status_rule)
        self.assertNotIn("#ecfdf3", status_rule.group("body"))

    def test_sidebar_does_not_scroll_with_conversation(self):
        body_rule = re.search(r"body\s*\{(?P<body>[^}]+)\}", self.html)
        chat_app_rule = re.search(r"\.chat-app\s*\{(?P<body>[^}]+)\}", self.html)
        chat_main_rule = re.search(r"\.chat-main\s*\{(?P<body>[^}]+)\}", self.html)
        conversation_rule = re.search(r"\.conversation\s*\{(?P<body>[^}]+)\}", self.html)

        self.assertIsNotNone(body_rule)
        self.assertIsNotNone(chat_app_rule)
        self.assertIsNotNone(chat_main_rule)
        self.assertIsNotNone(conversation_rule)
        self.assertIn("overflow: hidden", body_rule.group("body"))
        self.assertIn("height: 100dvh", chat_app_rule.group("body"))
        self.assertIn("overflow: hidden", chat_main_rule.group("body"))
        self.assertIn("overflow-y: auto", conversation_rule.group("body"))

    def test_chat_content_uses_balanced_width_like_modern_chat_apps(self):
        self.assertIn("--content-width: min(1120px, 100%)", self.html)
        conversation_rule = re.search(r"\.conversation\s*\{(?P<body>[^}]+)\}", self.html)
        assistant_rule = re.search(r"\.message\.assistant\s*\{(?P<body>[^}]+)\}", self.html)
        user_rule = re.search(r"\.message\.user\s*\{(?P<body>[^}]+)\}", self.html)
        row_rule = re.search(r"\.message-row\s*\{(?P<body>[^}]+)\}", self.html)
        prompt_rule = re.search(r"\.quick-prompts\s*\{(?P<body>[^}]+)\}", self.html)

        self.assertIsNotNone(conversation_rule)
        self.assertIsNotNone(assistant_rule)
        self.assertIsNotNone(user_rule)
        self.assertIsNotNone(row_rule)
        self.assertIsNotNone(prompt_rule)
        self.assertIn("width: 100%", conversation_rule.group("body"))
        self.assertIn("width: min(var(--content-width), 100%)", row_rule.group("body"))
        self.assertIn("width: min(860px, 100%)", assistant_rule.group("body"))
        self.assertIn("max-width: min(680px, 78%)", user_rule.group("body"))
        self.assertNotIn("max-width: 720px", prompt_rule.group("body"))

    def test_static_fake_conversation_is_removed(self):
        self.assertNotIn("最近一直拉肚子，还伴随腹痛，应该怎么办？", self.html)
        self.assertNotIn("Top 1 · 为什么总是泻肚", self.html)
        self.assertIn('data-empty-state="true"', self.html)

    def test_ui_is_wired_to_backend_health_and_ask_apis(self):
        self.assertIn("fetch('/api/health'", self.html)
        self.assertIn("fetch('/api/ask'", self.html)
        self.assertIn("currentMode", self.html)
        self.assertIn("renderAssistantMessage", self.html)

    def test_ui_supports_voice_input_and_local_history(self):
        self.assertIn("SpeechRecognition", self.html)
        self.assertIn("webkitSpeechRecognition", self.html)
        self.assertIn("localStorage", self.html)
        self.assertIn("renderHistory", self.html)

    def test_history_items_can_be_opened_and_deleted(self):
        self.assertIn("loadConversation", self.html)
        self.assertIn("deleteConversation", self.html)
        self.assertIn("history-delete", self.html)
        self.assertIn("currentConversationId", self.html)

    def test_api_settings_display_current_model(self):
        self.assertIn("online_model", self.html)
        self.assertIn("当前模型", self.html)
        self.assertIn("currentHealth", self.html)

    def test_voice_input_can_start_and_stop_listening(self):
        self.assertIn("currentRecognition", self.html)
        self.assertIn("stopSpeechRecognition", self.html)
        self.assertIn("aria-pressed", self.html)

    def test_loading_state_uses_spinner_indicator(self):
        self.assertIn("loading-spinner", self.html)
        self.assertIn("正在生成回答", self.html)
        self.assertIn("@keyframes spin", self.html)

    def test_history_delete_button_has_clear_label(self):
        self.assertIn("删除历史对话", self.html)
        self.assertIn(">删除</button>", self.html)

    def test_ui_reports_background_local_index_loading(self):
        self.assertIn("local_loading", self.html)
        self.assertIn("本地索引加载中", self.html)
        self.assertIn("本地索引尚未加载完成", self.html)
        self.assertIn("updateModeAvailability", self.html)

    def test_ui_renders_local_index_progress_bar(self):
        self.assertIn("index-progress", self.html)
        self.assertIn("index-progress-fill", self.html)
        self.assertIn("local_progress", self.html)
        self.assertIn("aria-valuenow", self.html)

    def test_ui_can_change_department_scope(self):
        self.assertIn("departmentScopes", self.html)
        self.assertIn("Pediatric_儿科", self.html)
        self.assertIn("fetch('/api/config'", self.html)
        self.assertIn("dataScopeSelect", self.html)

    def test_assistant_answer_supports_manual_voice_playback(self):
        self.assertIn("speechSynthesis", self.html)
        self.assertIn("SpeechSynthesisUtterance", self.html)
        self.assertIn("speak-button", self.html)
        self.assertIn("播放回答", self.html)

if __name__ == "__main__":
    unittest.main()
