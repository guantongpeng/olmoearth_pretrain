"""分词配置模块，用于自定义波段分组策略。

本模块允许为每种模态自定义波段（band）如何分组为 token，
支持不同的分词策略实验（如逐波段 token、光谱分组等）。

核心概念：
- band_group: 一组波段名称，作为一个 token 输入
- ModalityTokenization: 单个模态的分词配置
- TokenizationConfig: 全局分词配置，支持对特定模态的覆盖

使用示例：
    >>> from olmoearth_pretrain.nn.tokenization import TokenizationConfig, ModalityTokenization
    >>> from olmoearth_pretrain.data.constants import Modality
    >>>
    >>> # 为 Sentinel-2 创建逐波段分词配置
    >>> s2_bands = Modality.SENTINEL2_L2A.band_order
    >>> config = TokenizationConfig(
    ...     overrides={
    ...         Modality.SENTINEL2_L2A.name: ModalityTokenization(
    ...             band_groups=[[b] for b in s2_bands]
    ...         )
    ...     }
    ... )
    >>>
    >>> # 其他模态使用默认分词
    >>> num_bandsets = config.get_num_bandsets(Modality.SENTINEL1.name)
"""

from dataclasses import dataclass, field

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality, ModalitySpec


@dataclass
class ModalityTokenization(Config):
    """单个模态的自定义分词配置。

    指定波段如何分组为 token，每个 band_group 成为一个独立的 token。
    例如，Sentinel-2 有 12 个波段，可以：
    - 所有波段作为一组（1 个 token）
    - 每个波段作为一组（12 个 token）
    - 按光谱范围分组（如可见光、近红外、短波红外）

    Args:
        band_groups: 波段分组列表，每个分组是波段名称列表。
            例如 [["B04", "B03", "B02"], ["B08"], ["B11", "B12"]]
    """

    band_groups: list[list[str]]

    def compute_indices(self, base_modality: ModalitySpec) -> list[list[int]]:
        """Map band names to indices based on the base modality's band order.

        Args:
            base_modality: The ModalitySpec that defines the canonical band order.

        Returns:
            List of index lists, one per band group.

        Raises:
            ValueError: If a band name doesn't exist in the modality's band_order.
        """
        name_to_idx = {name: i for i, name in enumerate(base_modality.band_order)}
        result = []
        for group in self.band_groups:
            group_indices = []
            for band in group:
                if band not in name_to_idx:
                    raise ValueError(
                        f"Band '{band}' not found in modality '{base_modality.name}'. "
                        f"Valid bands: {list(base_modality.band_order)}"
                    )
                group_indices.append(name_to_idx[band])
            result.append(group_indices)
        return result

    def get_num_bands_per_group(self) -> list[int]:
        """Get the number of bands in each group."""
        return [len(group) for group in self.band_groups]

    @property
    def num_band_sets(self) -> int:
        """Get the number of band sets (token groups)."""
        return len(self.band_groups)

    def validate_against(self, base_modality: ModalitySpec) -> None:
        """Validate that all band names exist in the modality.

        Args:
            base_modality: The ModalitySpec to validate against.

        Raises:
            ValueError: If a band name doesn't exist in the modality's band_order.
        """
        valid_bands = set(base_modality.band_order)
        for group in self.band_groups:
            for band in group:
                if band not in valid_bands:
                    raise ValueError(
                        f"Band '{band}' not found in modality '{base_modality.name}'. "
                        f"Valid bands: {valid_bands}"
                    )


@dataclass
class TokenizationConfig(Config):
    """全局分词配置，支持对特定模态的覆盖。

    未被 overrides 覆盖的模态使用其 ModalitySpec 中的默认 bandset 配置。
    被覆盖的模态使用自定义的 ModalityTokenization 配置。

    关键属性：
        overrides: 模态名称到 ModalityTokenization 的映射字典
        _bandset_indices_cache: 波段集索引的缓存（避免重复计算）
    """

    overrides: dict[str, ModalityTokenization] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce raw dicts in overrides to ModalityTokenization instances."""
        self.overrides = {
            name: ModalityTokenization(**cfg) if isinstance(cfg, dict) else cfg
            for name, cfg in self.overrides.items()
        }

    _bandset_indices_cache: dict[str, list[list[int]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def get_bandset_indices(self, modality_name: str) -> list[list[int]]:
        """Get band indices for tokenization, using override or default.

        Args:
            modality_name: Name of the modality.

        Returns:
            List of index lists, one per bandset/token group.

        Raises:
            ValueError: If modality_name is invalid or band names don't exist.
        """
        # Check cache first
        if modality_name in self._bandset_indices_cache:
            return self._bandset_indices_cache[modality_name]

        try:
            base_spec = Modality.get(modality_name)
        except (AttributeError, AssertionError) as e:
            raise ValueError(f"Invalid modality: {modality_name}") from e

        if modality_name in self.overrides:
            result = self.overrides[modality_name].compute_indices(base_spec)
        else:
            result = base_spec.bandsets_as_indices()

        # Cache the result
        self._bandset_indices_cache[modality_name] = result
        return result

    def get_num_bandsets(self, modality_name: str) -> int:
        """Get number of bandsets (tokens per spatial location).

        Args:
            modality_name: Name of the modality.

        Returns:
            Number of bandsets.

        Raises:
            ValueError: If modality_name is invalid.
        """
        if modality_name in self.overrides:
            return self.overrides[modality_name].num_band_sets
        try:
            return Modality.get(modality_name).num_band_sets
        except (AttributeError, AssertionError) as e:
            raise ValueError(f"Invalid modality: {modality_name}") from e

    def get_num_bands_per_bandset(self, modality_name: str) -> list[int]:
        """Get the number of bands in each bandset.

        Args:
            modality_name: Name of the modality.

        Returns:
            List of band counts, one per bandset.

        Raises:
            ValueError: If modality_name is invalid.
        """
        if modality_name in self.overrides:
            return self.overrides[modality_name].get_num_bands_per_group()
        try:
            base_spec = Modality.get(modality_name)
        except (AttributeError, AssertionError) as e:
            raise ValueError(f"Invalid modality: {modality_name}") from e
        return [len(bs.bands) for bs in base_spec.band_sets]

    def validate(self) -> None:
        """Validate all overrides against their modalities.

        Raises:
            ValueError: If any modality name or band name is invalid.
        """
        for modality_name, tokenization in self.overrides.items():
            try:
                base_spec = Modality.get(modality_name)
            except (AttributeError, AssertionError):
                raise ValueError(
                    f"Invalid modality name in overrides: '{modality_name}'. "
                    f"Valid modalities: {Modality.names()}"
                )
            tokenization.validate_against(base_spec)
