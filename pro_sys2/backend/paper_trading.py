"""
نظام التداول الوهمي (Paper Trading)
=====================================
يسمح للمستخدمين بالتداول برصيد افتراضي لاختبار النظام
مع تتبع كامل للصفقات والأرباح والخسائر
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import threading

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class Trade(BaseModel):
    """صفقة تداول واحدة"""
    id: str
    user_id: str
    symbol: str
    type: str  # "BUY" or "SELL"
    quantity: float
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    status: str  # "OPEN" or "CLOSED"
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    signal_confidence: Optional[float] = None
    notes: Optional[str] = None


class Portfolio(BaseModel):
    """محفظة المستخدم"""
    user_id: str
    initial_balance: float
    current_balance: float
    total_profit_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_positions: List[Trade] = []
    closed_trades: List[Trade] = []
    created_at: str
    updated_at: str


# ─────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "paper_trading_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Thread-safe lock for file operations
storage_lock = threading.Lock()


def get_portfolio_path(user_id: str) -> str:
    """الحصول على مسار ملف المحفظة"""
    return os.path.join(DATA_DIR, f"portfolio_{user_id}.json")


def load_portfolio(user_id: str) -> Optional[Portfolio]:
    """تحميل محفظة المستخدم"""
    path = get_portfolio_path(user_id)
    
    with storage_lock:
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Portfolio(**data)
        except Exception as e:
            print(f"Error loading portfolio for {user_id}: {e}")
            return None


def save_portfolio(portfolio: Portfolio):
    """حفظ محفظة المستخدم"""
    path = get_portfolio_path(portfolio.user_id)
    portfolio.updated_at = datetime.utcnow().isoformat()
    
    with storage_lock:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(portfolio.dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving portfolio for {portfolio.user_id}: {e}")


def create_portfolio(user_id: str, initial_balance: float = 10000.0) -> Portfolio:
    """إنشاء محفظة جديدة"""
    portfolio = Portfolio(
        user_id=user_id,
        initial_balance=initial_balance,
        current_balance=initial_balance,
        total_profit_loss=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        open_positions=[],
        closed_trades=[],
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    save_portfolio(portfolio)
    return portfolio


# ─────────────────────────────────────────────
# Trading Operations
# ─────────────────────────────────────────────

def open_trade(user_id: str, symbol: str, trade_type: str, quantity: float, 
               entry_price: float, signal_confidence: float = None, notes: str = None) -> Dict:
    """فتح صفقة جديدة مع رسوم وانزلاق سعري واقعي"""
    portfolio = load_portfolio(user_id)
    
    # رسوم التداول (0.1% مثل Binance)
    TRADING_FEE = 0.001  # 0.1%
    
    # انزلاق سعري واقعي (0.02-0.05%)
    SLIPPAGE = random.uniform(0.0002, 0.0005)
    
    # السعر الفعلي مع الانزلاق
    if trade_type == 'BUY':
        actual_price = entry_price * (1 + SLIPPAGE)  # نشتري بسعر أعلى قليلاً
    else:
        actual_price = entry_price * (1 - SLIPPAGE)  # نبيع بسعر أقل قليلاً
    
    # حساب التكلفة الإجمالية مع الرسوم
    base_cost = quantity * actual_price
    fee = base_cost * TRADING_FEE
    total_cost = base_cost + fee
    
    # التحقق من الرصيد
    if total_cost > portfolio.current_balance:
        raise ValueError(f"رصيد غير كافٍ. المطلوب: ${total_cost:.2f}, المتاح: ${portfolio.current_balance:.2f}")
    
    # إنشاء الصفقة
    trade_id = f"trade_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    trade = Trade(
        id=trade_id,
        user_id=user_id,
        symbol=symbol,
        type=trade_type,
        quantity=quantity,
        entry_price=actual_price,  # السعر الفعلي بعد الانزلاق
        requested_price=entry_price,  # السعر المطلوب
        entry_time=datetime.now().isoformat(),
        status="OPEN",
        cost=base_cost,
        fee=fee,
        total_cost=total_cost,
        signal_confidence=signal_confidence,
        notes=notes
    )
    
    # تحديث الرصيد
    portfolio.current_balance -= total_cost
    portfolio.open_positions.append(trade)
    
    save_portfolio(portfolio)
    
    return {
        'success': True,
        'trade_id': trade_id,
        'actual_price': actual_price,
        'fee': fee,
        'slippage': (actual_price - entry_price) / entry_price * 100,
        'message': f'تم فتح صفقة {trade_type} لـ {quantity} {symbol} بسعر ${actual_price:.2f} (رسوم: ${fee:.2f})'
    }


def close_trade(user_id: str, trade_id: str, exit_price: float) -> Dict:
    """إغلاق صفقة مفتوحة مع رسوم وانزلاق واقعي"""
    portfolio = load_portfolio(user_id)
    
    # البحث عن الصفقة
    trade = None
    for i, t in enumerate(portfolio.open_positions):
        if t.id == trade_id:
            trade = portfolio.open_positions.pop(i)
            break
    
    if not trade:
        return {"success": False, "error": "الصفقة غير موجودة"}
    
    # رسوم التداول (0.1%)
    TRADING_FEE = 0.001
    
    # انزلاق سعري عند الإغلاق
    SLIPPAGE = random.uniform(0.0002, 0.0005)
    
    trade_type = trade.type
    
    # السعر الفعلي مع الانزلاق
    if trade_type == 'BUY':
        # عند إغلاق شراء (بيع)، نبيع بسعر أقل قليلاً
        actual_exit_price = exit_price * (1 - SLIPPAGE)
    else:
        # عند إغلاق بيع (شراء)، نشتري بسعر أعلى قليلاً
        actual_exit_price = exit_price * (1 + SLIPPAGE)
    
    # حساب القيمة والرسوم
    entry_price = trade.entry_price
    quantity = trade.quantity
    exit_value = quantity * actual_exit_price
    exit_fee = exit_value * TRADING_FEE
    net_exit_value = exit_value - exit_fee
    
    # حساب الربح/الخسارة الصافي (بعد خصم رسوم الدخول والخروج)
    entry_fee = trade.fee
    
    if trade_type == 'BUY':
        # ربح = (سعر الخروج - سعر الدخول) × الكمية - الرسوم
        gross_profit = (actual_exit_price - entry_price) * quantity
        net_profit = gross_profit - entry_fee - exit_fee
    else:  # SELL
        # ربح = (سعر الدخول - سعر الخروج) × الكمية - الرسوم
        gross_profit = (entry_price - actual_exit_price) * quantity
        net_profit = gross_profit - entry_fee - exit_fee
    
    profit_loss_pct = (net_profit / trade.cost) * 100
    
    # تحديث الصفقة
    trade.exit_price = actual_exit_price
    trade.requested_exit_price = exit_price
    trade.exit_time = datetime.utcnow().isoformat()
    trade.exit_fee = exit_fee
    trade.gross_profit_loss = gross_profit
    trade.profit_loss = net_profit
    trade.profit_loss_pct = profit_loss_pct
    trade.status = "CLOSED"
    trade.total_fees = entry_fee + exit_fee
    
    # تحديث الرصيد (نضيف القيمة الصافية بعد الرسوم)
    portfolio.current_balance += net_exit_value
    portfolio.total_profit_loss += net_profit
    
    # نقل الصفقة إلى السجل
    portfolio.closed_trades.append(trade)
    
    save_portfolio(portfolio)
    
    return {
        "success": True,
        "profit_loss": net_profit,
        "profit_loss_pct": profit_loss_pct,
        "actual_exit_price": actual_exit_price,
        "exit_fee": exit_fee,
        "total_fees": entry_fee + exit_fee,
        "slippage": (actual_exit_price - exit_price) / exit_price * 100,
        "message": f'تم إغلاق الصفقة. {"ربح" if net_profit >= 0 else "خسارة"}: ${abs(net_profit):.2f} ({profit_loss_pct:+.2f}%) | رسوم: ${entry_fee + exit_fee:.2f}'
    }


def get_portfolio_summary(user_id: str) -> Dict:
    """الحصول على ملخص المحفظة"""
    
    portfolio = load_portfolio(user_id)
    if not portfolio:
        portfolio = create_portfolio(user_id)
    
    # Calculate win rate
    win_rate = 0.0
    if portfolio.total_trades > 0:
        win_rate = (portfolio.winning_trades / portfolio.total_trades) * 100
    
    # Calculate total portfolio value (balance + open positions)
    open_positions_value = sum(
        pos.quantity * pos.entry_price for pos in portfolio.open_positions
    )
    total_value = portfolio.current_balance + open_positions_value
    
    # Calculate ROI
    roi = ((total_value - portfolio.initial_balance) / portfolio.initial_balance) * 100
    
    return {
        "user_id": user_id,
        "initial_balance": portfolio.initial_balance,
        "current_balance": portfolio.current_balance,
        "open_positions_value": open_positions_value,
        "total_value": total_value,
        "total_profit_loss": portfolio.total_profit_loss,
        "roi": roi,
        "total_trades": portfolio.total_trades,
        "winning_trades": portfolio.winning_trades,
        "losing_trades": portfolio.losing_trades,
        "win_rate": win_rate,
        "open_positions_count": len(portfolio.open_positions),
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at
    }


def get_trade_history(user_id: str, limit: int = 50) -> Dict:
    """الحصول على سجل الصفقات"""
    
    portfolio = load_portfolio(user_id)
    if not portfolio:
        return {
            "open_positions": [],
            "closed_trades": []
        }
    
    # Sort closed trades by exit time (most recent first)
    closed_trades = sorted(
        portfolio.closed_trades,
        key=lambda t: t.exit_time or "",
        reverse=True
    )[:limit]
    
    return {
        "open_positions": [t.dict() for t in portfolio.open_positions],
        "closed_trades": [t.dict() for t in closed_trades]
    }


def reset_portfolio(user_id: str, initial_balance: float = 10000.0) -> Dict:
    """إعادة تعيين المحفظة"""
    portfolio = create_portfolio(user_id, initial_balance)
    return {
        "success": True,
        "message": "تم إعادة تعيين المحفظة بنجاح",
        "portfolio": get_portfolio_summary(user_id)
    }


def get_performance_stats(user_id: str) -> Dict:
    """إحصائيات الأداء التفصيلية"""
    
    portfolio = load_portfolio(user_id)
    if not portfolio or len(portfolio.closed_trades) == 0:
        return {
            "total_trades": 0,
            "avg_profit_per_trade": 0,
            "avg_loss_per_trade": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "profit_factor": 0
        }
    
    winning_trades = [t for t in portfolio.closed_trades if t.profit_loss > 0]
    losing_trades = [t for t in portfolio.closed_trades if t.profit_loss <= 0]
    
    avg_profit = sum(t.profit_loss for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.profit_loss for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    largest_win = max((t.profit_loss for t in winning_trades), default=0)
    largest_loss = min((t.profit_loss for t in losing_trades), default=0)
    
    avg_win_pct = sum(t.profit_loss_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss_pct = sum(t.profit_loss_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    total_wins = sum(t.profit_loss for t in winning_trades)
    total_losses = abs(sum(t.profit_loss for t in losing_trades))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    
    return {
        "total_trades": len(portfolio.closed_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "avg_profit_per_trade": avg_profit,
        "avg_loss_per_trade": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
        "total_profit": total_wins,
        "total_loss": total_losses
    }
