import re
from loguru import logger
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ace_step.language_segmentation import LangSegment
from ace_step.ace_models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer

SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, 
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293, 
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}
lyric_tokenizer = VoiceBpeTokenizer()
lang_segment = LangSegment()
lang_segment.langid.set_languages(list(SUPPORT_LANGUAGES.keys()))
lang_segment.setfilters([
            'af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
            'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
            'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
            'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
            'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
            'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'
        ])


def get_lang(text, default_lang, threshold):
    language = "en"
    try:
        words = lang_segment.getTexts(text)
        print(words)
        if words[0]['score'] < threshold:
            return default_lang
        langCounts = lang_segment.getCounts()
        language = langCounts[0][0]
        if len(langCounts) > 1 and language == "en":
            language = langCounts[1][0]
    except Exception as err:
        language = "en"
    return language


def tokenize_lyrics(lyrics, default_lang, threshold):
    lines = lyrics.split("\n")
    lyric_token_idx = []
    for line in lines:
        line = line.strip()
        if not line:
            lyric_token_idx += ["\n"]
            continue

        lang = get_lang(line, default_lang, threshold)

        if lang not in SUPPORT_LANGUAGES:
            lang = "en"
        if "zh" in lang:
            lang = "zh"
        if "spa" in lang:
            lang = "es"

        structure_pattern = re.compile(r"\[.*?\]")
        try:
            if structure_pattern.match(line):
                lyric_token_idx.append(line + "\n")
            else:
                token_idx = lyric_tokenizer.preprocess_text(line, lang)
                lyric_token_idx.append(f"[{lang}]" + token_idx + "\n")
        except Exception as e:
            print("tokenize error", e, "for line", line, "major_language", lang)

    return "".join(lyric_token_idx)


class LyricsLangSwitch:
    # Mapping of language codes to display names
    LANGUAGE_MAP = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "cs": "Czech",
        "nl": "Dutch",
        "ar": "Arabic",
        "zh": "Chinese",
        "ja": "Japanese",
        "hu": "Hungarian",
        "ko": "Korean",
        "hi": "Hindi"
    }

    # Create a list of display names for the dropdown
    LANGUAGE_DISPLAY_NAMES = sorted(list(LANGUAGE_MAP.values()))

    # Create a reverse mapping to easily get the code from the display name
    # This is efficient for lookup in the lyricsgen method
    REVERSE_LANGUAGE_MAP = {v: k for k, v in LANGUAGE_MAP.items()}

    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True, 
                    "default": "[verse]\n我走过深夜的街道\n冷风吹乱思念的漂亮外套\n你的微笑像星光很炫耀\n照亮了我孤独的每分每秒\n\n[chorus]\n愿你是风吹过我的脸\n带我飞过最远最遥远的山间\n愿你是风轻触我的梦\n停在心头不再飘散无迹无踪\n\n[verse]\n一起在喧哗避开世俗的骚动\n独自在天台探望月色的朦胧\n你说爱像音乐带点重节奏\n一拍一跳让我忘了心的温度多空洞\n\n[bridge]\n唱起对你的想念不隐藏\n像诗又像画写满藏不了的渴望\n你的影子挥不掉像风的倔强\n追着你飞扬穿越云海一样泛光\n\n[chorus]\n愿你是风吹过我的手\n暖暖的触碰像春日细雨温柔\n愿你是风盘绕我的身\n深情万万重不会有一天走远走\n\n[verse]\n深夜的钢琴弹起动人的旋律\n低音鼓砸进心底的每一次呼吸\n要是能将爱化作歌声传递\n你是否会听见我心里的真心实意"}),
                "language": (cls.LANGUAGE_DISPLAY_NAMES, {"default": cls.LANGUAGE_MAP["en"]}),
                "threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,       # Minimum value
                    "max": 1.0,       # Maximum value
                    "step": 0.01,     # Step for the slider/input
                    "display": "number"}),  # or "slider"
            },
        }

    CATEGORY = "🎤MW/MW-ACE-Step"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "lyricsgen"
    
    def lyricsgen(self, lyrics: str, language: str, threshold: float):
        # Convert the selected display name back to the language code
        language_code = self.REVERSE_LANGUAGE_MAP.get(language)
        if language_code is None:
            # Fallback or error handling if the display name isn't found
            language_code = "en" # Default to 'en' if something goes wrong
        return (tokenize_lyrics(lyrics.strip(), language_code, threshold),)
