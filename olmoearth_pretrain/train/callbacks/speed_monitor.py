"""
速度监控回调模块。

本模块提供 OlmoEarthSpeedMonitorCallback，用于在训练过程中监控和记录：
- 编码器吞吐量（TPS Encoded）：每秒处理的编码 token 数
- 解码器吞吐量（TPS Decoded）：每秒处理的解码 token 数
- 目标编码器吞吐量（TPS Target Encoder）
- 批次吞吐量（BPS）：每秒处理的批次数
- 数据加载时间占比
- 模型前向/反向传播时间占比

这些指标有助于诊断训练瓶颈（数据加载 vs 模型计算）。
"""

"""Speed monitor callback for the trainer for OlmoEarth Pretrain."""
import time
from typing import Any

from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
)
from olmoearth_pretrain.train.train_module.galileo import GalileoTrainModule
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModule
from olmoearth_pretrain.train.train_module.mae import MAETrainModule

logger = logging.getLogger(__name__)


class OlmoEarthSpeedMonitorCallback(SpeedMonitorCallback):
    """OlmoEarth 训练速度监控回调。

    继承自 olmo-core 的 SpeedMonitorCallback，扩展了以下功能：
    - 根据 token_budget、encode_ratio、decode_ratio 计算真实的 token 吞吐量
    - 分别追踪编码器、解码器、目标编码器的 token 处理量
    - 支持不同训练模块（MAE、LatentMIM、ContrastiveLatentMIM、Galileo）

    关键属性:
        priority: 回调优先级（10，数值越小越先执行）
        _total_tokens_encoded: 累计编码 token 数
        _total_tokens_decoded: 累计解码 token 数
        _total_tokens_target_encoder: 累计目标编码器 token 数
    """

    priority = 10  # 回调优先级
    _total_tokens_encoded = 0  # 累计编码 token 数
    _total_tokens_decoded = 0  # 累计解码 token 数
    _total_tokens_target_encoder = 0  # 累计目标编码器 token 数

    def pre_train(self) -> None:
        """训练开始前的回调：初始化速度监控参数。

        根据训练模块类型获取 encode_ratio 和 decode_ratio，
        用于后续计算每个步骤的 token 吞吐量。对于 Galileo 训练模块，
        使用 masking_strategy_a 的比率，若 strategy_b 比率不同则发出警告。
        """
        super().pre_train()
        train_module = self.trainer.train_module

        self._token_budget = self.trainer.data_loader.token_budget  # 每个样本的 token 预算
        if isinstance(
            train_module,
            MAETrainModule | LatentMIMTrainModule | ContrastiveLatentMIMTrainModule,
        ):
            # 对于 MAE / LatentMIM / ContrastiveLatentMIM 模块，使用其掩码策略的编解码比率
            self._encoder_ratio = train_module.masking_strategy.encode_ratio
            self._decoder_ratio = train_module.masking_strategy.decode_ratio
            logger.warning(
                "Speed monitor callback bases token input based on token budget, "
                "encoder ratio, and decoder ratio"
            )
        elif isinstance(train_module, GalileoTrainModule):
            # 对于 Galileo 模块，使用 masking_strategy_a 的比率
            self._encoder_ratio = train_module.masking_strategy_a.encode_ratio
            self._decoder_ratio = train_module.masking_strategy_a.decode_ratio
            # 若 strategy_b 的比率与 strategy_a 不同，发出警告
            if train_module.masking_strategy_b.encode_ratio != self._encoder_ratio:
                logger.warning(
                    "Speed monitor callback bases token input based on encoder ratio "
                    "from masking_strategy_a"
                )
            if train_module.masking_strategy_b.decode_ratio != self._decoder_ratio:
                logger.warning(
                    "Speed monitor callback bases token input based on decoder ratio "
                    "from masking_strategy_a"
                )
            logger.warning(
                "Speed monitor callback bases token input based on token budget, "
                "encoder ratio, and decoder ratio"
            )
        else:
            # 不支持的训练模块类型，仅计算基本吞吐量
            logger.warning(
                "Speed monitor callback only calculates token throughput with "
                "MAETrainModule, LatentMIMTrainModule or GalileoTrainModule"
            )

    def pre_load_batch(self) -> None:
        """批次加载前的回调：记录回调间隔时间。

        测量从上一步的 post_step 结束到当前 pre_load_batch 开始之间的时间，
        用于计算回调本身的开销时间。
        """
        if hasattr(self, "callback_start_time"):
            self.callback_start_time: float
            # 假设此回调是第一个执行的，测量回调间隔时间
            self.trainer.record_metric(
                "throughput/callback time (s)",
                time.perf_counter() - self.callback_start_time,
            )
        super().pre_load_batch()

    def pre_step(self, batch: Any) -> None:
        """训练步进前的回调：计算当前步的 token 数量并记录。

        根据批次大小、编码比率、解码比率和 token 预算计算当前步的
        编码/解码/目标编码器 token 数量，并累加到总计中。
        跳过第一步，因为第一步通常耗时异常。

        Args:
            batch: 当前批次数据，可以是2元组(patch_size, sample)或3元组(patch_size, sample_a, sample_b)
        """
        self._batch_load_time = time.perf_counter() - self._batch_load_start  # 计算数据加载耗时
        if self._first_step:
            # 跳过第一步，因为第一步通常耗时异常长（包含初始化等开销）
            return
        # 从批次中获取样本数据（支持2元组和3元组格式）
        sample: MaskedOlmoEarthSample = batch[1]
        # 计算当前步的 token 数量
        self._step_tokens_encoded = (
            sample.batch_size * self._encoder_ratio * self._token_budget
        )  # 编码 token 数 = 批次大小 × 编码比率 × token预算
        self._step_tokens_decoded = (
            sample.batch_size * self._decoder_ratio * self._token_budget
        )  # 解码 token 数 = 批次大小 × 解码比率 × token预算
        self._step_tokens_target_encoder = sample.batch_size * self._token_budget  # 目标编码器 token 数

        # 累加到总计
        self._total_steps += 1
        self._total_tokens_encoded += self._step_tokens_encoded
        self._total_tokens_decoded += self._step_tokens_decoded
        self._total_tokens_target_encoder += self._step_tokens_target_encoder
        self.model_start_time = time.perf_counter()  # 记录模型计算开始时间

    def post_step(self) -> None:
        """训练步进后的回调：计算并记录各种吞吐量指标。

        计算并记录以下指标：
        - 编码/解码/目标编码器的 token 吞吐量（瞬时和平均）
        - 批次吞吐量 BPS（瞬时和平均）
        - 数据加载时间占比
        - 模型计算时间占比
        """
        counter = time.perf_counter()
        self.model_end_time = counter  # 记录模型计算结束时间

        # 记录数据加载时间
        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time
        )
        self._first_step: bool
        if self._first_step:
            # 第一步完成后初始化计时器，开始正式记录
            self._total_steps = 0
            self._total_tokens = 0
            self._start_time = counter
            self._step_last_logged = counter
            self._first_step = False
            return

        # 计算各种时间指标
        self.model_duration = self.model_end_time - self.model_start_time  # 模型计算耗时
        step_time = counter - self._step_last_logged  # 步骤总耗时
        total_time = counter - self._start_time  # 从开始到现在的总耗时
        self._step_last_logged = counter

        # 计算批次吞吐量（Batches Per Second）
        bps = 1 / step_time  # 瞬时 BPS
        bps_avg = self._total_steps / total_time  # 平均 BPS
        # 保存平均 BPS 供 beaker 回调使用（估算剩余时间）
        self._bps_avg = bps_avg

        # 计算各组件吞吐量和占比
        data_pct = 100 * self._batch_load_time / step_time  # 数据加载时间占比(%)
        tps_encoded = self._total_tokens_encoded / step_time  # 瞬时编码 TPS
        tps_encoded_avg = self._total_tokens_encoded / total_time  # 平均编码 TPS
        tps_decoded = self._total_tokens_decoded / step_time  # 瞬时解码 TPS
        tps_decoded_avg = self._total_tokens_decoded / total_time  # 平均解码 TPS
        tps_target_encoder = self._total_tokens_target_encoder / step_time  # 瞬时目标编码器 TPS
        tps_target_encoder_avg = self._total_tokens_target_encoder / total_time  # 平均目标编码器 TPS

        # 记录所有吞吐量指标
        self.trainer.record_metric(
            "throughput/total tokens target encoder-since-restart",
            self._total_tokens_target_encoder,
        )

        self.trainer.record_metric(
            "throughput/total tokens encoded-since-restart", self._total_tokens_encoded
        )
        self.trainer.record_metric(
            "throughput/total tokens decoded-since-restart", self._total_tokens_decoded
        )
        self.trainer.record_metric("throughput/device/TPS Encoded", tps_encoded)
        self.trainer.record_metric(
            "throughput/device/TPS Target Encoder", tps_target_encoder
        )
        self.trainer.record_metric(
            "throughput/device/TPS Target Encoder (estimated avg)",
            tps_target_encoder_avg,
        )
        self.trainer.record_metric(
            "throughput/device/TPS Encoded (estimated avg)", tps_encoded_avg
        )
        self.trainer.record_metric("throughput/device/TPS Decoded", tps_decoded)
        self.trainer.record_metric(
            "throughput/device/TPS Decoded (estimated avg)", tps_decoded_avg
        )
        self.trainer.record_metric("throughput/device/data loading (%)", data_pct)
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (estimated avg)", bps_avg)
        self.trainer.record_metric(
            "throughput/device/model duration (s)", self.model_duration
        )
        self.trainer.record_metric(
            "throughput/device/model duration (%)", self.model_duration / step_time
        )
        self.callback_start_time = time.perf_counter()


HeliosSpeedMonitorCallback = _deprecated_class_alias(
    OlmoEarthSpeedMonitorCallback,
    "helios.train.callbacks.speed_monitor.HeliosSpeedMonitorCallback",
)
