# enhanced_scoring_system.py
# 增强版动态评分系统 - 支持形态共振和自适应权重

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logger_utils import Colors, print_colored


class EnhancedScoringSystem:
    """
    增强版评分系统
    特点：
    1. 动态权重调整
    2. 形态共振检测
    3. 市场环境自适应
    4. 多时间框架确认
    """

    def __init__(self, logger=None):
        self.logger = logger

        # 基础评分权重（会根据市场环境动态调整）
        self.base_weights = {
            'categories': {
                'trend': 0.25,
                'technical': 0.30,
                'game_theory': 0.25,
                'market_structure': 0.20
            },
            'patterns': {
                # 反转形态权重
                'head_shoulders': 0.85,
                'inverse_head_shoulders': 0.90,
                'double_top': 0.75,
                'double_bottom': 0.80,

                # 持续形态权重
                'triangle': 0.65,
                'flag': 0.70,
                'wedge': 0.60,

                # 博弈论形态权重
                'stop_hunt': 0.80,
                'liquidity_grab': 0.85,
                'wyckoff': 0.90,
                'squeeze': 0.85,

                # 市场结构权重
                'poc_reversion': 0.75,
                'value_area_trade': 0.70,
                'trend_day': 0.80
            }
        }

        # 市场环境配置
        self.market_regimes = {
            'TRENDING': {'trend': 0.40, 'technical': 0.25, 'game_theory': 0.20, 'market_structure': 0.15},
            'RANGING': {'trend': 0.15, 'technical': 0.25, 'game_theory': 0.30, 'market_structure': 0.30},
            'VOLATILE': {'trend': 0.20, 'technical': 0.20, 'game_theory': 0.35, 'market_structure': 0.25},
            'BREAKOUT': {'trend': 0.30, 'technical': 0.30, 'game_theory': 0.25, 'market_structure': 0.15}
        }

        # 形态共振配置
        self.resonance_groups = {
            'reversal': ['head_shoulders', 'double_top', 'double_bottom', 'wyckoff_spring'],
            'continuation': ['flag', 'triangle', 'wedge'],
            'manipulation': ['stop_hunt', 'liquidity_grab', 'false_breakout'],
            'squeeze': ['short_squeeze', 'long_squeeze', 'exhaustion']
        }

        print_colored("✅ 增强版评分系统初始化完成", Colors.GREEN)

    def calculate_comprehensive_score(self, analysis_data: Dict) -> Dict:
        """计算综合评分"""
        # 1. 检测市场环境
        market_regime = self._detect_market_regime(analysis_data)

        # 2. 调整权重
        adjusted_weights = self._adjust_weights_for_regime(market_regime)

        # 3. 计算各类别得分
        category_scores = self._calculate_category_scores(analysis_data)

        # 4. 检测形态共振
        resonance_bonus = self._detect_pattern_resonance(analysis_data)

        # 5. 多时间框架确认
        mtf_multiplier = self._calculate_mtf_multiplier(analysis_data)

        # 6. 计算最终得分
        final_score = self._compute_final_score(
            category_scores,
            adjusted_weights,
            resonance_bonus,
            mtf_multiplier
        )

        # 7. 生成详细报告
        report = self._generate_score_report(
            final_score,
            category_scores,
            adjusted_weights,
            resonance_bonus,
            mtf_multiplier,
            market_regime
        )

        return report

    def _detect_market_regime(self, analysis_data: Dict) -> str:
        """检测当前市场环境"""
        indicators = analysis_data.get('technical_indicators', {})
        market_profile = analysis_data.get('market_profile', {})

        # 获取关键指标
        adx = indicators.get('ADX', 25)
        atr_ratio = indicators.get('ATR_ratio', 1.0)
        volume_trend = indicators.get('volume_trend', 1.0)
        bb_width = indicators.get('bb_width', 0.02)

        # 趋势持续性加分
        duration = trend_data.get('duration', 0)
        if duration > 20:  # 持续20个周期以上
            score *= 1.2
        elif duration < 5:  # 新趋势减分
            score *= 0.8

        # 趋势质量
        quality = trend_data.get('quality', 0.5)
        score *= quality

        return max(-10, min(10, score))

    def _calculate_technical_score(self, tech_data: Dict) -> float:
        """计算技术指标得分（-10到10）"""
        score = 0

        # RSI评分（趋势感知）
        rsi = tech_data.get('RSI', 50)
        trend_direction = tech_data.get('trend_direction', 'NEUTRAL')

        if trend_direction == 'UP':
            # 上升趋势中，RSI 40-80 都是正常的
            if 40 <= rsi <= 60:
                score += 2.0
            elif 60 < rsi <= 80:
                score += 1.0
            elif rsi < 40:
                score += 3.0  # 上升趋势中超卖是机会
            elif rsi > 80:
                score -= 1.0  # 轻微超买
        elif trend_direction == 'DOWN':
            # 下降趋势中，RSI 20-60 都是正常的
            if 40 <= rsi <= 60:
                score -= 2.0
            elif 20 <= rsi < 40:
                score -= 1.0
            elif rsi > 60:
                score -= 3.0  # 下降趋势中超买是做空机会
            elif rsi < 20:
                score += 1.0  # 极度超卖可能反弹
        else:
            # 震荡市场使用传统解读
            if rsi < 30:
                score += 2.5
            elif rsi > 70:
                score -= 2.5

        # MACD评分
        macd = tech_data.get('MACD', 0)
        macd_signal = tech_data.get('MACD_signal', 0)
        macd_histogram = tech_data.get('MACD_histogram', 0)

        if macd > macd_signal:
            score += 1.5
            if macd_histogram > 0 and tech_data.get('MACD_histogram_increasing', False):
                score += 1.0  # 柱状图增长
        else:
            score -= 1.5
            if macd_histogram < 0 and tech_data.get('MACD_histogram_decreasing', False):
                score -= 1.0

        # 布林带位置
        bb_position = tech_data.get('bb_position', 50)
        if bb_position < 20:
            score += 2.0
        elif bb_position > 80:
            score -= 2.0

        # 成交量确认
        volume_ratio = tech_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score *= 1.3  # 放量确认
        elif volume_ratio < 0.5:
            score *= 0.7  # 缩量减分

        # 其他指标
        # CCI
        cci = tech_data.get('CCI', 0)
        if cci > 100:
            score += 1.0
        elif cci < -100:
            score -= 1.0

        # Williams %R
        williams_r = tech_data.get('Williams_R', -50)
        if williams_r > -20:
            score -= 1.0  # 超买
        elif williams_r < -80:
            score += 1.0  # 超卖

        return max(-10, min(10, score))

    def _calculate_game_theory_score(self, patterns: List[Dict]) -> float:
        """计算博弈论形态得分"""
        score = 0

        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            confidence = pattern.get('confidence', 0)
            direction = pattern.get('direction', 'NEUTRAL')

            # 获取形态权重
            weight = self.base_weights['patterns'].get(pattern_type, 0.5)

            # 计算单个形态得分
            pattern_score = confidence * weight * 5  # 最高5分每个形态

            # 根据方向调整
            if direction == 'BULLISH':
                score += pattern_score
            elif direction == 'BEARISH':
                score -= pattern_score

        return max(-10, min(10, score))

    def _calculate_market_structure_score(self, market_data: Dict) -> float:
        """计算市场结构得分"""
        score = 0

        # POC相关
        poc_distance = market_data.get('poc_distance', 0)
        if poc_distance > 0.01:  # 偏离POC超过1%
            score += 2.0 * min(poc_distance * 100, 1)  # 最多2分

        # 价值区域位置
        va_position = market_data.get('value_area_position', 'IN_VALUE')
        if va_position == 'ABOVE_VALUE':
            score -= 1.5  # 在价值区域上方，可能回落
        elif va_position == 'BELOW_VALUE':
            score += 1.5  # 在价值区域下方，可能反弹

        # 市场阶段
        market_phase = market_data.get('phase', 'BALANCE')
        if market_phase == 'TRENDING':
            trend_direction = market_data.get('trend_direction', 0)
            score += trend_direction * 3  # 趋势市场加强信号
        elif market_phase == 'BALANCE':
            score *= 0.5  # 平衡市场减弱信号

        # 拍卖状态
        auction_state = market_data.get('auction_state', '')
        if auction_state == 'INITIATIVE_BUYING':
            score += 2.0
        elif auction_state == 'INITIATIVE_SELLING':
            score -= 2.0

        return max(-10, min(10, score))

    def _detect_pattern_resonance(self, analysis_data: Dict) -> float:
        """检测形态共振"""
        resonance_bonus = 0

        # 收集所有检测到的形态
        all_patterns = []

        # 技术形态
        if 'classical_patterns' in analysis_data:
            all_patterns.extend(analysis_data['classical_patterns'])

        # 博弈论形态
        if 'game_theory_patterns' in analysis_data:
            all_patterns.extend(analysis_data['game_theory_patterns'])

        # 检查每个共振组
        for group_name, group_patterns in self.resonance_groups.items():
            group_count = 0
            group_direction = None

            for pattern in all_patterns:
                if pattern['type'] in group_patterns:
                    group_count += 1
                    if group_direction is None:
                        group_direction = pattern['direction']
                    elif group_direction != pattern['direction']:
                        group_count = 0  # 方向不一致，不算共振
                        break

            # 计算共振加分
            if group_count >= 2:
                if group_name == 'reversal':
                    resonance_bonus += 2.0
                elif group_name == 'manipulation':
                    resonance_bonus += 1.5
                elif group_name == 'continuation':
                    resonance_bonus += 1.0
                elif group_name == 'squeeze':
                    resonance_bonus += 1.8

        # 跨类别共振检测
        tech_signals = analysis_data.get('technical_indicators', {}).get('signal_direction', 0)
        game_signals = sum(1 if p['direction'] == 'BULLISH' else -1
                           for p in analysis_data.get('game_theory_patterns', []))

        if tech_signals * game_signals > 0:  # 同向
            resonance_bonus += 1.0

        return resonance_bonus

    def _calculate_mtf_multiplier(self, analysis_data: Dict) -> float:
        """计算多时间框架确认乘数"""
        mtf_data = analysis_data.get('multi_timeframe', {})

        if not mtf_data:
            return 1.0

        # 计算各时间框架的信号方向
        timeframe_signals = []
        timeframe_weights = {
            '5m': 0.1,
            '15m': 0.15,
            '1h': 0.25,
            '4h': 0.25,
            '1d': 0.25
        }

        agreement_score = 0
        total_weight = 0

        for tf, data in mtf_data.items():
            if tf in timeframe_weights:
                signal = data.get('signal', 0)  # 1=买入, -1=卖出, 0=中性
                weight = timeframe_weights[tf]

                if signal != 0:
                    timeframe_signals.append(signal)
                    agreement_score += signal * weight
                    total_weight += weight

        # 计算一致性
        if len(timeframe_signals) >= 3:
            # 检查是否所有信号同向
            if all(s > 0 for s in timeframe_signals) or all(s < 0 for s in timeframe_signals):
                return 1.5  # 完全一致，加成50%
            elif abs(agreement_score / total_weight) > 0.6:
                return 1.3  # 大部分一致，加成30%
            elif abs(agreement_score / total_weight) > 0.3:
                return 1.1  # 轻微一致，加成10%

        return 1.0

    def _compute_final_score(self, category_scores: Dict, weights: Dict,
                             resonance_bonus: float, mtf_multiplier: float) -> Dict:
        """计算最终得分"""
        # 基础加权得分
        weighted_score = sum(category_scores[cat] * weights[cat]
                             for cat in category_scores)

        # 应用共振加成
        enhanced_score = weighted_score + resonance_bonus

        # 应用多时间框架乘数
        final_score = enhanced_score * mtf_multiplier

        # 计算置信度（0-1）
        confidence = min(abs(final_score) / 10, 0.95)

        # 确定交易方向和强度
        if final_score > 2.0:
            action = 'BUY'
            if final_score > 4.0:
                action = 'STRONG_BUY'
        elif final_score < -2.0:
            action = 'SELL'
            if final_score < -4.0:
                action = 'STRONG_SELL'
        else:
            action = 'HOLD'

        return {
            'final_score': final_score,
            'action': action,
            'confidence': confidence,
            'weighted_score': weighted_score,
            'resonance_bonus': resonance_bonus,
            'mtf_multiplier': mtf_multiplier
        }

    def _generate_score_report(self, final_score: Dict, category_scores: Dict,
                               weights: Dict, resonance_bonus: float,
                               mtf_multiplier: float, market_regime: str) -> Dict:
        """生成详细评分报告"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'market_regime': market_regime,
            'final_score': final_score['final_score'],
            'action': final_score['action'],
            'confidence': final_score['confidence'],
            'details': {
                'category_scores': category_scores,
                'adjusted_weights': weights,
                'resonance_bonus': resonance_bonus,
                'mtf_multiplier': mtf_multiplier,
                'weighted_score': final_score['weighted_score']
            },
            'breakdown': self._generate_score_breakdown(
                category_scores, weights, resonance_bonus, mtf_multiplier
            )
        }

        # 添加交易建议
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _generate_score_breakdown(self, scores: Dict, weights: Dict,
                                  resonance: float, mtf: float) -> List[str]:
        """生成评分细分说明"""
        breakdown = []

        # 各类别贡献
        for category, score in scores.items():
            contribution = score * weights[category]
            breakdown.append(
                f"{category.capitalize()}: {score:.2f} × {weights[category]:.2f} = {contribution:.2f}"
            )

        # 特殊加成
        if resonance > 0:
            breakdown.append(f"形态共振加成: +{resonance:.2f}")

        if mtf != 1.0:
            breakdown.append(f"多时间框架乘数: ×{mtf:.2f}")

        return breakdown

    def _generate_recommendations(self, report: Dict) -> Dict:
        """生成交易建议"""
        recommendations = {
            'position_size': 0,
            'risk_level': 'MEDIUM',
            'entry_strategy': '',
            'exit_strategy': '',
            'warnings': []
        }

        confidence = report['confidence']
        action = report['action']

        # 仓位建议
        if confidence > 0.8:
            recommendations['position_size'] = 0.03  # 3%仓位
            recommendations['risk_level'] = 'LOW'
        elif confidence > 0.6:
            recommendations['position_size'] = 0.02  # 2%仓位
            recommendations['risk_level'] = 'MEDIUM'
        else:
            recommendations['position_size'] = 0.01  # 1%仓位
            recommendations['risk_level'] = 'HIGH'

        # 入场策略
        if 'STRONG' in action:
            recommendations['entry_strategy'] = '市价入场'
        else:
            recommendations['entry_strategy'] = '限价入场，等待回调'

        # 出场策略
        if report['market_regime'] == 'TRENDING':
            recommendations['exit_strategy'] = '移动止损，让利润奔跑'
        elif report['market_regime'] == 'RANGING':
            recommendations['exit_strategy'] = '固定止盈，快进快出'
        else:
            recommendations['exit_strategy'] = '严格止损，保守止盈'

        # 风险警告
        if report['market_regime'] == 'VOLATILE':
            recommendations['warnings'].append('市场波动较大，注意控制仓位')

        if confidence < 0.6:
            recommendations['warnings'].append('信号置信度较低，建议观望或减小仓位')

        return recommendations势强度
        if adx > 35:
            return 'TRENDING'
        elif adx < 20:
            # 低ADX + 窄布林带 = 区间震荡
            if bb_width < 0.015:
                return 'RANGING'
            # 低ADX + 宽布林带 = 高波动
            elif bb_width > 0.04:
                return 'VOLATILE'

        # 检查是否即将突破
        if bb_width < 0.01 and volume_trend > 1.3:
            return 'BREAKOUT'

        # 默认
        if atr_ratio > 1.5:
            return 'VOLATILE'
        else:
            return 'RANGING'

    def _adjust_weights_for_regime(self, regime: str) -> Dict:
        """根据市场环境调整权重"""
        return self.market_regimes.get(regime, self.base_weights['categories'])

    def _calculate_category_scores(self, analysis_data: Dict) -> Dict:
        """计算各类别得分"""
        scores = {
            'trend': 0,
            'technical': 0,
            'game_theory': 0,
            'market_structure': 0
        }

        # 1. 趋势得分
        trend_data = analysis_data.get('trend', {})
        if trend_data:
            scores['trend'] = self._calculate_trend_score(trend_data)

        # 2. 技术指标得分
        tech_data = analysis_data.get('technical_indicators', {})
        if tech_data:
            scores['technical'] = self._calculate_technical_score(tech_data)

        # 3. 博弈论得分
        game_patterns = analysis_data.get('game_theory_patterns', [])
        if game_patterns:
            scores['game_theory'] = self._calculate_game_theory_score(game_patterns)

        # 4. 市场结构得分
        market_data = analysis_data.get('market_auction', {})
        if market_data:
            scores['market_structure'] = self._calculate_market_structure_score(market_data)

        return scores

    def _calculate_trend_score(self, trend_data: Dict) -> float:
        """计算趋势得分（-10到10）"""
        score = 0

        # 趋势方向和强度
        direction = trend_data.get('direction', 'NEUTRAL')
        strength = trend_data.get('strength', 0)

        if direction == 'UP':
            score = strength * 10
        elif direction == 'DOWN':
            score = -strength * 10

        # 趋