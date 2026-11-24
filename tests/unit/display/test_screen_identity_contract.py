"""Contract tests for screen identity, primary flag and hotplug policy.

These tests exercise the backend-independent :class:`ScreenRegistry` with
faked native snapshots.  They cover the four required scenarios:

* primary + secondary screens,
* duplicate monitor names,
* disconnect / reconnect (including changed volatile handles),
* an empty display list.
"""
import pytest

from pyglet.display.identity import (
    ScreenInfo,
    ScreenRegistry,
    UNKNOWN_MONITOR_NAME,
    assign_primary_fallback,
    default_key,
)


def info(stable, name="Monitor", *, native=None, x=0, y=0, w=1920, h=1080,
         primary=False, rate=60.0, hardware=None, native_stable=False,
         backend="fake", ordinal=0):
    return ScreenInfo(
        backend=backend, ordinal=ordinal, x=x, y=y, width=w, height=h,
        is_primary=primary, refresh_rate=rate, native_id=native,
        native_id_is_stable=native_stable, stable_native_id=stable,
        hardware_id=hardware, name=name,
    )


@pytest.mark.unit
class TestPrimaryAndSecondary:
    def test_primary_lookup_and_order(self):
        registry = ScreenRegistry("fake")
        primary = info("OUT-1", primary=True, x=0, y=0)
        secondary = info("OUT-2", primary=False, x=1920, y=0)

        identities = registry.synchronize([primary, secondary])

        assert [i.stable_native_id for i in identities] == ["OUT-1", "OUT-2"]
        assert registry.primary.key == identities[0].key
        assert registry.primary.is_primary is True
        assert identities[1].is_primary is False
        assert len(registry) == 2

    def test_primary_fallback_to_origin_screen(self):
        # No native primary flag: the (0, 0) screen becomes primary.
        left = info("OUT-1", x=-1920, y=0)
        origin = info("OUT-2", x=0, y=0)
        assign_primary_fallback([left, origin])
        assert origin.is_primary is True
        assert left.is_primary is False

        registry = ScreenRegistry("fake")
        identities = registry.synchronize([left, origin])
        assert registry.primary.key == identities[1].key

    def test_primary_fallback_to_first_when_no_origin(self):
        first = info("OUT-1", x=100, y=100)
        second = info("OUT-2", x=3000, y=100)
        assign_primary_fallback([first, second])
        assert first.is_primary is True

    def test_existing_primary_flag_is_not_overwritten(self):
        weird_origin = info("OUT-1", primary=False, x=0, y=0)
        flagged = info("OUT-2", primary=True, x=999, y=999)
        assign_primary_fallback([weird_origin, flagged])
        assert flagged.is_primary is True
        assert weird_origin.is_primary is False


@pytest.mark.unit
class TestDuplicateNames:
    def test_same_name_gets_distinct_identities_and_suffix_labels(self):
        registry = ScreenRegistry("fake")
        a = info("OUT-1", name="DELL U2719D", native=11)
        b = info("OUT-2", name="DELL U2719D", native=22, x=1920)

        ids = registry.synchronize([a, b])

        assert ids[0].key != ids[1].key
        assert ids[0].label == "DELL U2719D"
        assert ids[1].label == "DELL U2719D #2"
        # Raw name is preserved and remains duplicated.
        assert ids[0].name == ids[1].name == "DELL U2719D"

    def test_duplicate_suffix_survives_reenumeration(self):
        registry = ScreenRegistry("fake")
        a = info("OUT-1", name="Same")
        b = info("OUT-2", name="Same", x=1920)
        first = registry.synchronize([a, b])

        # Re-enumerate in the opposite order: identity follows the output,
        # not the list position.
        second = registry.synchronize([b, a])
        by_out = {i.stable_native_id: i for i in second}
        assert by_out["OUT-1"].key == first[0].key
        assert by_out["OUT-2"].key == first[1].key
        labels = sorted(i.label for i in second)
        assert labels == ["Same", "Same #2"]

    def test_three_identical_screens(self):
        registry = ScreenRegistry("fake")
        infos = [info(f"OUT-{i}", name="Twin", x=1920 * i) for i in range(3)]
        ids = registry.synchronize(infos)
        assert len({i.key for i in ids}) == 3
        assert sorted(i.label for i in ids) == ["Twin", "Twin #2", "Twin #3"]


@pytest.mark.unit
class TestDisconnectReconnect:
    def test_disconnect_marks_absent_and_resolve_returns_none(self):
        registry = ScreenRegistry("fake")
        k1, k2 = [i.key for i in registry.synchronize([
            info("OUT-1", primary=True), info("OUT-2", x=1920),
        ])]

        registry.synchronize([info("OUT-1", primary=True)])

        assert len(registry) == 1
        assert registry.resolve(k1) is not None
        assert registry.resolve(k2) is None
        # History is retained for reconciliation.
        assert registry.get(k2).connected is False

    def test_reconnect_same_stable_token_restores_key(self):
        registry = ScreenRegistry("fake")
        key_b = registry.synchronize([
            info("OUT-1", primary=True), info("OUT-2", x=1920),
        ])[1].key

        registry.synchronize([info("OUT-1", primary=True)])
        restored = registry.synchronize([
            info("OUT-1", primary=True), info("OUT-2", x=1920),
        ])
        assert restored[1].key == key_b
        assert registry.resolve(key_b) is not None

    def test_reconnect_with_changed_volatile_handle(self):
        # Win32-like: HMONITOR handle changes after replug but the GDI
        # device name stays the same.
        registry = ScreenRegistry("win32")
        first = registry.synchronize([
            info(r"\\.\DISPLAY1", backend="win32", native=0xA001),
            info(r"\\.\DISPLAY2", backend="win32", native=0xB002, x=1920),
        ])
        key2 = first[1].key

        registry.synchronize([
            info(r"\\.\DISPLAY1", backend="win32", native=0xA009),
        ])
        reconnected = registry.synchronize([
            info(r"\\.\DISPLAY1", backend="win32", native=0xA009),
            info(r"\\.\DISPLAY2", backend="win32", native=0xC777, x=1920),
        ])
        assert reconnected[1].key == key2

    def test_reconnect_by_hardware_id_across_port_change(self):
        registry = ScreenRegistry("fake")
        old_key = registry.synchronize([
            info("DP-1", hardware="SN-123", name="Dell"),
        ])[0].key

        # Same physical panel moved to a different connector.
        new = registry.synchronize([
            info("HDMI-2", hardware="SN-123", name="Dell"),
        ])
        assert new[0].key == old_key

    def test_unambiguous_fingerprint_reconnect(self):
        # No stable token/native id: unique name+resolution bridges.
        registry = ScreenRegistry("fake")
        old_key = registry.synchronize([
            info(stable=None, name="OnlyOne", w=1920, h=1080),
        ])[0].key

        registry.synchronize([])
        new = registry.synchronize([
            info(stable=None, name="OnlyOne", w=1920, h=1080),
        ])
        assert new[0].key == old_key

    def test_ambiguous_identical_twins_are_not_guessed(self):
        # Two indistinguishable same-name screens with no stable token.
        # After one snapshot disappears and a (possibly different) twin
        # returns, the registry must not silently rebind the missing key.
        registry = ScreenRegistry("fake")
        ids = registry.synchronize([
            info(stable=None, name="Twin", x=0),
            info(stable=None, name="Twin", x=1920),
        ])
        key_a, key_b = ids[0].key, ids[1].key

        registry.synchronize([
            info(stable=None, name="Twin", x=0),
        ])
        # A twin reconnects.  Because the fingerprint is ambiguous in
        # history, matching must not arbitrarily claim key_b for it while
        # key_a also exists.  Re-syncing both yields two unique identities.
        again = registry.synchronize([
            info(stable=None, name="Twin", x=0),
            info(stable=None, name="Twin", x=1920),
        ])
        assert len({i.key for i in again}) == 2
        assert key_a in registry
        assert key_b in registry

    def test_replacement_monitor_gets_new_identity(self):
        registry = ScreenRegistry("fake")
        old_key = registry.synchronize([info("OUT-2", name="Old")])[0].key
        registry.synchronize([])

        new = registry.synchronize([info("OUT-9", name="New")])
        assert new[0].key != old_key
        assert registry.get(old_key).connected is False
        assert registry.resolve(old_key) is None


@pytest.mark.unit
class TestEmptyDisplayList:
    def test_synchronize_empty(self):
        registry = ScreenRegistry("fake")
        registry.synchronize([info("OUT-1")])

        result = registry.synchronize([])

        assert result == []
        assert registry.screens == []
        assert registry.primary is None
        assert len(registry) == 0

    def test_empty_without_prior_snapshot(self):
        registry = ScreenRegistry("fake")
        assert registry.synchronize(None) == []
        assert registry.primary is None

    def test_reconnect_after_empty_restores_identity(self):
        registry = ScreenRegistry("fake")
        key = registry.synchronize([info("OUT-1")])[0].key
        registry.synchronize([])

        restored = registry.synchronize([info("OUT-1")])
        assert restored[0].key == key
        assert registry.resolve(key) is restored[0]


@pytest.mark.unit
class TestFallbacksAndDeterminism:
    def test_missing_name_is_unknown(self):
        registry = ScreenRegistry("fake")
        identity = registry.synchronize([info("OUT-1", name=None)])[0]
        assert identity.name == UNKNOWN_MONITOR_NAME
        assert identity.label == UNKNOWN_MONITOR_NAME

    def test_geometry_fallback_key_is_deterministic(self):
        a = ScreenInfo(backend="x11", x=0, y=0, width=1920, height=1080)
        b = ScreenInfo(backend="x11", x=0, y=0, width=1920, height=1080)
        assert default_key(a) == default_key(b)
        # Unlike Python's salted builtin hash, the key is a stable string.
        assert isinstance(default_key(a), str)
        assert default_key(a).startswith("x11:fp:")

    def test_invalid_refresh_dropped_at_info_boundary(self):
        record = info("OUT-1", rate=0)
        assert record.refresh_rate is None
