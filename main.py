import os
import re
import time
import secrets
import logging
import asyncio
import httpx
from dataclasses import dataclass
from typing import Optional, List

import asyncpg
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# =========================================================
# CONFIG
# =========================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

DB_HOST = os.getenv("DB_HOST", "localhost").strip()
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASS = os.getenv("DB_PASS", "").strip()

ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()]

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is required")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
log = logging.getLogger("Mokafih")

# ======================
# Hugging Face config
# ======================
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL")

log.info("HF_API_KEY set=%s HF_MODEL=%s", bool(HF_API_KEY), HF_MODEL)


async def aitana_score(text: str) -> float:
    """
    Returns fraud likelihood between 0.0 and 1.0 using Hugging Face Inference API.
    Fail-safe: never crashes, never blocks for too long.
    """
    # If env vars are missing, do nothing
    if not HF_API_KEY or not HF_MODEL:
        log.warning("HF missing: key=%r model=%r", HF_API_KEY, HF_MODEL)
        return 0.0

    payload = {"inputs": text[:4000]}
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                f"https://router.huggingface.co/models/{HF_MODEL}",
                headers=headers,
                json=payload,
            )

        log.warning("HF status=%s body=%s", r.status_code, r.text[:400])

        if r.status_code != 200:
            return 0.0

        data = r.json()
        # Standard text-classification response is a list of {label, score} objects.
        if not isinstance(data, list) or not data:
            return 0.0

        # Pick the label with highest score
        best = max(data, key=lambda x: x.get("score", 0.0))
        label = str(best.get("label", "")).lower()
        score = float(best.get("score", 0.0))

        log.warning("HF best label=%s score=%.3f", label, score)

        # Map your model's labels to "fraud probability".
        # Adjust these mappings based on the actual labels returned by your model.
        fraud_like_labels = {"fraud", "scam", "malicious", "label_1"}
        if label in fraud_like_labels:
            return score

        # Example: if using a negative/positive sentiment model
        if label in {"negative", "neg"}:
            return score

        return 0.0

    except Exception as e:
        log.exception("HF error: %s", e)
        return 0.0


# =========================================================
# STRINGS
# =========================================================
STRINGS = {
    "en": {
        "report": "🛡️ *SECURITY REPORT*",
        "high": "🟥 *High Caution Recommended*",
        "medium": "🟧 *Proceed Carefully*",
        "low": "🟩 *No Immediate Concern*",
        "confidence": "Overall confidence:",
        "likely": "🔍 *Likely scenario:*",
        "indicators": "📊 *Key indicators:*",
        "actions": "✅ *Recommended actions:*",
        "safe": "✅ This was safe",
        "scam": "🚫 This was a scam",
        "thanks": "Thanks — feedback recorded.",
        "lang_set_en": "✅ Language set to English.",
        "lang_set_ar": "✅ تم تغيير اللغة إلى العربية.",
        "welcome": "Send any message and I'll assess risk. Use /login to activate. Use /language to switch.",
        "need_login": "🔒 Access denied. Activate with `/login KEY`.",
        "expired": "❌ License expired. Contact your administrator.",
        "license_full": "⛔ License seats are full.",
        "login_ok": "✅ Activated for",
        "invalid_key": "❌ Invalid key.",
        "expired_key": "❌ Key expired.",
        "usage_login": "Usage: /login KEY",
        "usage_genkey": "Usage: /genkey COMPANY SEATS DAYS",
        "admin_only": "Admin only.",
        "db_down": "⚠️ Backend unavailable. Cannot verify license right now.",
        "whoami_admin": "Admin",
        "whoami_guest": "Guest (unlicensed)",
        "whoami_user": "Licensed user",
    },
    "ar": {
        "report": "🛡️ *تقرير أمني*",
        "high": "🟥 *يوصى بحذر شديد*",
        "medium": "🟧 *تعامل بحذر*",
        "low": "🟩 *لا يوجد قلق فوري*",
        "confidence": "مستوى الثقة:",
        "likely": "🔍 *السيناريو المرجح:*",
        "indicators": "📊 *مؤشرات الخطورة:*",
        "actions": "✅ *إجراءات مقترحة:*",
        "safe": "✅ كانت آمنة",
        "scam": "🚫 كانت احتيال",
        "thanks": "شكرًا — تم تسجيل التقييم.",
        "lang_set_en": "✅ Language set to English.",
        "lang_set_ar": "✅ تم تغيير اللغة إلى العربية.",
        "welcome": "أرسل أي رسالة وسأقيّم الخطورة. فعّل عبر /login. غيّر اللغة عبر /language.",
        "need_login": "🔒 تم رفض الوصول. فعّل باستخدام `/login KEY`.",
        "expired": "❌ انتهت صلاحية الرخصة. تواصل مع المسؤول.",
        "license_full": "⛔ المقاعد ممتلئة في هذه الرخصة.",
        "login_ok": "✅ تم التفعيل لصالح",
        "invalid_key": "❌ مفتاح غير صالح.",
        "expired_key": "❌ المفتاح منتهي.",
        "usage_login": "الاستخدام: /login KEY",
        "usage_genkey": "الاستخدام: /genkey COMPANY SEATS DAYS",
        "admin_only": "للأدمن فقط.",
        "db_down": "⚠️ النظام الخلفي غير متاح. لا يمكن التحقق من الرخصة الآن.",
        "whoami_admin": "أدمن",
        "whoami_guest": "ضيف (غير مفعّل)",
        "whoami_user": "مستخدم مفعّل",
    },
}

FLAG_TEXT = {
    "urgency": {"en": "Uses urgent / pressure language", "ar": "أسلوب استعجال وضغط"},
    "sensitive": {"en": "Requests sensitive information", "ar": "طلب معلومات حساسة"},
    "context": {"en": "Mentions bank / payment / delivery", "ar": "سياق بنكي أو دفع أو توصيل"},
    "url": {"en": "Contains external link", "ar": "يحتوي على رابط خارجي"},
    "login_domain": {"en": "Link resembles login / verification page", "ar": "الرابط يشبه صفحة تسجيل أو تحقق"},
    "mixed": {"en": "Mixed-language social engineering", "ar": "هندسة اجتماعية بلغتين"},
}

# =========================================================
# DATABASE (timeouts + migrations)
# =========================================================
class Database:
    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None

    async def init(self) -> None:
        self.pool = await asyncpg.create_pool(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            min_size=1,
            max_size=5,
            timeout=5,
            command_timeout=5,
        )
        async with self.pool.acquire() as c:
            await c.execute("""
                CREATE TABLE IF NOT EXISTS licenses (
                    key TEXT PRIMARY KEY,
                    company TEXT NOT NULL,
                    max_seats INTEGER NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                )
            """)
            await c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    license_key TEXT REFERENCES licenses(key),
                    login_at DOUBLE PRECISION,
                    language TEXT DEFAULT 'en'
                )
            """)
            await c.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    user_id BIGINT,
                    label TEXT,
                    created_at DOUBLE PRECISION NOT NULL
                )
            """)
        log.info("PostgreSQL ready")

    async def get_lang(self, user_id: int) -> str:
        if not self.pool:
            return "en"
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT language FROM users WHERE user_id=$1", user_id)
            if row and row["language"] in ("en", "ar"):
                return row["language"]
        return "en"

    async def set_lang(self, user_id: int, lang: str) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as c:
            await c.execute("""
                INSERT INTO users (user_id, language, login_at)
                VALUES ($1, $2, 0)
                ON CONFLICT (user_id) DO UPDATE SET language = EXCLUDED.language
            """, user_id, lang)

    async def get_user_with_license(self, user_id: int) -> Optional[asyncpg.Record]:
        if not self.pool:
            return None
        async with self.pool.acquire() as c:
            return await c.fetchrow("""
                SELECT u.user_id, u.license_key, u.login_at, u.language,
                       l.company, l.max_seats, l.expires_at, l.status
                FROM users u
                LEFT JOIN licenses l ON u.license_key = l.key
                WHERE u.user_id = $1
            """, user_id)

    async def create_license(self, key: str, company: str, seats: int, days: int) -> None:
        if not self.pool:
            return
        expires_at = time.time() + (days * 86400)
        async with self.pool.acquire() as c:
            await c.execute("""
                INSERT INTO licenses (key, company, max_seats, expires_at, created_at, status)
                VALUES ($1, $2, $3, $4, $5, 'active')
                ON CONFLICT (key) DO UPDATE SET
                    company = EXCLUDED.company,
                    max_seats = EXCLUDED.max_seats,
                    expires_at = EXCLUDED.expires_at,
                    status = 'active'
            """, key, company, seats, expires_at, time.time())

    async def get_license(self, key: str) -> Optional[asyncpg.Record]:
        if not self.pool:
            return None
        async with self.pool.acquire() as c:
            return await c.fetchrow("SELECT * FROM licenses WHERE key=$1", key)

    async def seat_count(self, key: str) -> int:
        if not self.pool:
            return 0
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT COUNT(*) AS cnt FROM users WHERE license_key=$1", key)
            return int(row["cnt"])

    async def login_user(self, user_id: int, key: str) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as c:
            await c.execute("""
                INSERT INTO users (user_id, license_key, login_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO UPDATE SET
                    license_key = EXCLUDED.license_key,
                    login_at = EXCLUDED.login_at
            """, user_id, key, time.time())

    async def save_analysis(self, aid: str, user_id: int) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as c:
            await c.execute("""
                INSERT INTO analyses (id, user_id, created_at)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
            """, aid, user_id, time.time())

    async def set_label(self, aid: str, label: str) -> bool:
        if not self.pool:
            return False
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT label FROM analyses WHERE id=$1", aid)
            if not row or row["label"] is not None:
                return False
            await c.execute("UPDATE analyses SET label=$2 WHERE id=$1", aid, label)
            return True


db = Database()

# =========================================================
# ANALYSIS (Primary/Secondary/Possible, no percentages)
# =========================================================
AR_URGENCY = ["عاجل", "فور", "خلال", "تم إيقاف", "مهلة", "الآن"]
AR_SENSITIVE = ["رمز", "كلمة المرور", "تسجيل", "OTP", "كود", "تحقق", "تأكيد"]
AR_CONTEXT = ["بنك", "دفع", "فاتورة", "شحنة", "توصيل", "جمارك", "محفظة", "بطاقة"]

EN_URGENCY = ["urgent", "within", "suspended", "final notice", "action required"]
EN_SENSITIVE = ["verify", "login", "password", "otp", "code", "confirm"]
EN_CONTEXT = ["bank", "payment", "delivery", "account", "invoice", "wallet"]

URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)


@dataclass
class AnalysisResult:
    posture: str
    signals: List[str]
    hypotheses: List[str]


def analyze(text: str) -> AnalysisResult:
    t = text.lower()
    signals = set()

    ar = bool(re.search(r"[\u0600-\u06FF]", text))
    en = bool(re.search(r"[a-zA-Z]", text))

    if ar and any(w in text for w in AR_URGENCY):
        signals.add("urgency")
    if ar and any(w in text for w in AR_SENSITIVE):
        signals.add("sensitive")
    if ar and any(w in text for w in AR_CONTEXT):
        signals.add("context")

    if en and any(w in t for w in EN_URGENCY):
        signals.add("urgency")
    if en and any(w in t for w in EN_SENSITIVE):
        signals.add("sensitive")
    if en and any(w in t for w in EN_CONTEXT):
        signals.add("context")

    if ar and en:
        signals.add("mixed")

    urls = URL_RE.findall(text[:4000])
    if urls:
        signals.add("url")
        for u in urls:
            if any(x in u.lower() for x in ["login", "verify", "secure", "bank", "update"]):
                signals.add("login_domain")

    if {"urgency", "sensitive", "url"} <= signals:
        posture = "high"
    elif {"url", "urgency"} <= signals or {"url", "sensitive"} <= signals:
        posture = "medium"
    elif "url" in signals:
        posture = "medium"
    else:
        posture = "low"

    hypotheses: List[str] = []
    if "sensitive" in signals:
        hypotheses.append("Account takeover attempt" if not ar else "محاولة استيلاء على حساب")
    if "urgency" in signals:
        hypotheses.append("Social engineering pressure" if not ar else "ضغط هندسة اجتماعية")
    if "url" in signals:
        hypotheses.append("Generic phishing attempt" if not ar else "محاولة تصيّد عامة")
    if "context" in signals:
        hypotheses.append("Delivery / payment scam" if not ar else "احتيال توصيل/دفع")

    return AnalysisResult(
        posture=posture,
        signals=sorted(list(signals)),
        hypotheses=hypotheses[:3],
    )


# =========================================================
# AI + HEURISTIC FUSION (Aitana)
# =========================================================
async def analyze_with_ai(text: str) -> AnalysisResult:
    """
    Wraps analyze() and allows AI to ESCALATE risk only.
    """
    res = analyze(text)

    ai_score = await aitana_score(text)
    log.warning("AITANA | score=%.3f | text=%r", ai_score, text[:80])

    # AI can only escalate, never downgrade
    if ai_score >= 0.85:
        res.posture = "high"
        if "AI-detected fraud pattern" not in res.hypotheses:
            res.hypotheses.insert(0, "AI-detected fraud pattern")
    elif ai_score >= 0.60 and res.posture == "low":
        res.posture = "medium"
        res.hypotheses.insert(0, "AI-detected suspicious pattern")

    return res


# =========================================================
# UI
# =========================================================
def feedback_kb(aid: str, lang: str) -> InlineKeyboardMarkup:
    S = STRINGS[lang]
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(S["safe"], callback_data=f"fb:{aid}:safe"),
            InlineKeyboardButton(S["scam"], callback_data=f"fb:{aid}:scam"),
        ]
    ])


def build_report(res: AnalysisResult, lang: str) -> str:
    S = STRINGS[lang]
    lines = [
        S["report"],
        "",
        S[res.posture],
        f"{S['confidence']} *{res.posture.upper()}*",
        "",
        S["likely"],
    ]

    if res.hypotheses:
        if len(res.hypotheses) >= 1:
            lines.append(f"• Primary: {res.hypotheses[0]}")
        if len(res.hypotheses) >= 2:
            lines.append(f"• Secondary: {res.hypotheses[1]}")
        if len(res.hypotheses) >= 3:
            lines.append(f"• Possible: {res.hypotheses[2]}")

    lines += ["", S["indicators"]]
    for s in res.signals:
        lines.append(f"• {FLAG_TEXT.get(s, {}).get(lang, s)}")

    lines += [
        "",
        S["actions"],
    ]
    if lang == "ar":
        lines += [
            "• لا تضغط على الروابط.",
            "• لا تشارك كلمات المرور أو رموز OTP.",
            "• تحقق عبر التطبيق/الموقع الرسمي (ادخل يدويًا).",
        ]
    else:
        lines += [
            "• Do not click links.",
            "• Do not share passwords or OTP codes.",
            "• Verify via the official app/website (type it manually).",
        ]

    return "\n".join(lines)


# =========================================================
# LICENSING ENFORCEMENT
# =========================================================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


async def require_license_or_reject(update: Update, lang: str) -> bool:
    """
    Returns True if allowed, False if rejected (reply already sent).
    Hard enforcement.
    """
    user_id = update.effective_user.id
    if is_admin(user_id):
        return True

    S = STRINGS[lang]

    try:
        user = await db.get_user_with_license(user_id)
    except Exception as e:
        log.error("DB error get_user_with_license: %s", e)
        await update.effective_message.reply_text(S["db_down"])
        return False

    if not user or not user.get("license_key"):
        await update.effective_message.reply_text(S["need_login"], parse_mode=ParseMode.MARKDOWN)
        return False

    # license row missing or inactive
    if user.get("status") != "active" or user.get("expires_at") is None:
        await update.effective_message.reply_text(S["need_login"], parse_mode=ParseMode.MARKDOWN)
        return False

    if float(user["expires_at"]) < time.time():
        await update.effective_message.reply_text(S["expired"])
        return False

    return True


# =========================================================
# COMMANDS
# =========================================================
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        lang = await db.get_lang(update.effective_user.id)
    except Exception:
        lang = "en"
    await update.effective_message.reply_text(STRINGS[lang]["welcome"])


async def cmd_language(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    try:
        current = await db.get_lang(user_id)
        new_lang = "ar" if current == "en" else "en"
        await db.set_lang(user_id, new_lang)
    except Exception as e:
        log.error("DB error set_lang: %s", e)
        new_lang = "en"
    await update.effective_message.reply_text(
        STRINGS[new_lang]["lang_set_ar"] if new_lang == "ar" else STRINGS[new_lang]["lang_set_en"]
    )


async def cmd_genkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    lang = "en"
    try:
        lang = await db.get_lang(user_id)
    except Exception:
        pass

    S = STRINGS[lang]
    if not is_admin(user_id):
        await update.effective_message.reply_text(S["admin_only"])
        return

    try:
        company = ctx.args[0].strip()
        seats = int(ctx.args[1])
        days = int(ctx.args[2])
        if seats <= 0 or days <= 0:
            raise ValueError("invalid")
    except Exception:
        await update.effective_message.reply_text(S["usage_genkey"])
        return

    key = f"MOK-{company[:3].upper()}-{secrets.token_hex(3).upper()}"
    try:
        await db.create_license(key, company, seats, days)
    except Exception as e:
        log.error("DB error create_license: %s", e)
        await update.effective_message.reply_text(S["db_down"])
        return

    await update.effective_message.reply_text(f"✅ Key: `{key}`", parse_mode=ParseMode.MARKDOWN)


async def cmd_login(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    lang = "en"
    try:
        lang = await db.get_lang(user_id)
    except Exception:
        pass
    S = STRINGS[lang]

    if not ctx.args:
        await update.effective_message.reply_text(S["usage_login"])
        return

    key = ctx.args[0].strip()

    try:
        lic = await db.get_license(key)
    except Exception as e:
        log.error("DB error get_license: %s", e)
        await update.effective_message.reply_text(S["db_down"])
        return

    if not lic or lic.get("status") != "active":
        await update.effective_message.reply_text(S["invalid_key"])
        return

    if float(lic["expires_at"]) < time.time():
        await update.effective_message.reply_text(S["expired_key"])
        return

    # Seat enforcement
    try:
        used = await db.seat_count(key)
        if used >= int(lic["max_seats"]):
            current = await db.get_user_with_license(user_id)
            if not current or current.get("license_key") != key:
                await update.effective_message.reply_text(S["license_full"])
                return
        await db.login_user(user_id, key)
    except Exception as e:
        log.error("DB error login_user/seat_count: %s", e)
        await update.effective_message.reply_text(S["db_down"])
        return

    await update.effective_message.reply_text(f"{S['login_ok']} {lic['company']}")


async def cmd_whoami(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    lang = "en"
    try:
        lang = await db.get_lang(user_id)
    except Exception:
        pass
    S = STRINGS[lang]

    if is_admin(user_id):
        await update.effective_message.reply_text(f"{S['whoami_admin']}\nID: `{user_id}`", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        u = await db.get_user_with_license(user_id)
    except Exception:
        await update.effective_message.reply_text(S["db_down"])
        return

    if not u or not u.get("license_key") or u.get("expires_at") is None:
        await update.effective_message.reply_text(S["whoami_guest"])
        return

    days_left = max(0, int((float(u["expires_at"]) - time.time()) / 86400))
    await update.effective_message.reply_text(
        f"{S['whoami_user']}\n🏢 {u.get('company')}\n🔑 {u.get('license_key')}\n⏳ {days_left} days"
    )


# =========================================================
# HANDLERS
# =========================================================
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    try:
        lang = await db.get_lang(user_id)
    except Exception:
        lang = "en"

    # Hard license enforcement
    allowed = await require_license_or_reject(update, lang)
    if not allowed:
        return

    res = await analyze_with_ai(update.message.text)
    aid = secrets.token_hex(6)

    try:
        await db.save_analysis(aid, user_id)
    except Exception as e:
        log.warning("save_analysis failed: %s", e)

    await update.message.reply_text(
        build_report(res, lang),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
        reply_markup=feedback_kb(aid, lang),
    )


async def handle_feedback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q or not q.data:
        return

    try:
        await q.answer()
    except Exception:
        pass

    parts = q.data.split(":")
    if len(parts) != 3 or parts[0] != "fb":
        return

    aid, label = parts[1], parts[2]
    if label not in ("safe", "scam"):
        return

    user_id = q.from_user.id
    try:
        lang = await db.get_lang(user_id)
    except Exception:
        lang = "en"
    S = STRINGS[lang]

    try:
        ok = await db.set_label(aid, label)
    except Exception as e:
        log.warning("set_label failed: %s", e)
        ok = False

    try:
        await q.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    if ok:
        try:
            await q.message.reply_text(S["thanks"])
        except Exception:
            pass


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error: %s", context.error)


# =========================================================
# STARTUP (retry DB init)
# =========================================================
async def startup(app: Application) -> None:
    for attempt in range(10):
        try:
            await db.init()
            return
        except Exception as e:
            log.warning("DB not ready (attempt %d/10): %s", attempt + 1, e)
            await asyncio.sleep(2)
    log.error("DB never became available. Bot will run but licensing will fail closed.")


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(startup)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("language", cmd_language))
    app.add_handler(CommandHandler("genkey", cmd_genkey))
    app.add_handler(CommandHandler("login", cmd_login))
    app.add_handler(CommandHandler("whoami", cmd_whoami))

    app.add_handler(CallbackQueryHandler(handle_feedback, pattern=r"^fb:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.add_error_handler(on_error)

    log.info("Mokafih running (polling).")
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)


if __name__ == "__main__":
    main()
