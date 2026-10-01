# Verbatim excerpt of sglang 0.5.20 python/sglang/srt/mem_cache/unified_radix_cache.py (Apache-2.0),
# UnifiedRadixCache.inc_lock_ref / dec_lock_ref, for auditkit.compat rule (i).
class UnifiedRadixCache:
    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node_id)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_lock_ref(node_id, skip_lock_components)

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: DecLockRefParams,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node_id, params)
        if result is not None:
