"""Взаимное согласие между людьми: таблица `contacts` и `ContactStore` (F-602)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hub import migrations_runner
from hub.contacts import ContactError, ContactStore, canonical_pair

AMY = "p-amy"
MAX = "p-max"
KAI = "p-kai"


@pytest.fixture
def store(tmp_path):
    conn = migrations_runner.connect(str(tmp_path / "hub.db"))
    migrations_runner.migrate(conn)
    for person_id, name in ((AMY, "Amy"), (MAX, "Max"), (KAI, "Kai")):
        conn.execute("INSERT INTO persons(person_id, display_name) VALUES (?,?)",
                     (person_id, name))
    conn.commit()
    yield ContactStore(conn), conn
    conn.close()


class TestCanonicalPair:
    def test_the_same_pair_in_any_order_is_one_row(self):
        assert canonical_pair(MAX, AMY) == canonical_pair(AMY, MAX) == (AMY, MAX)

    def test_a_person_cannot_be_their_own_contact(self):
        with pytest.raises(ContactError):
            canonical_pair(AMY, AMY)

    def test_a_blank_name_is_refused(self):
        with pytest.raises(ContactError):
            canonical_pair(AMY, "  ")


class TestInvitation:
    def test_inviting_writes_a_pending_row_into_the_real_table(self, store):
        contacts, conn = store
        contact = contacts.invite(AMY, MAX)
        assert contact.status == "pending" and contact.requested_by == AMY
        assert not contact.confirmed and contact.confirmed_at is None
        row = conn.execute(
            "SELECT person_a, person_b, status, requested_by FROM contacts").fetchone()
        assert tuple(row) == (AMY, MAX, "pending", AMY)
        assert contacts.status(MAX, AMY) == "pending"

    def test_repeating_the_invitation_changes_nothing(self, store):
        contacts, conn = store
        first = contacts.invite(AMY, MAX)
        second = contacts.invite(MAX, AMY)
        assert first.created_at == second.created_at
        assert second.requested_by == AMY, "the first invitation still stands"
        assert len(list(conn.execute("SELECT 1 FROM contacts"))) == 1

    def test_the_invited_person_can_say_yes(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contact = contacts.confirm(MAX, AMY, now=datetime(2026, 9, 22, 12, tzinfo=UTC))
        assert contact.status == "accepted" and contact.confirmed
        assert contact.confirmed_at == datetime(2026, 9, 22, 12, tzinfo=UTC)
        assert contacts.contact_gate(AMY, MAX).confirmed
        assert contacts.contact_gate(MAX, AMY).confirmed

    def test_a_person_cannot_confirm_their_own_invitation(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        with pytest.raises(ContactError):
            contacts.confirm(AMY, MAX)
        assert contacts.status(AMY, MAX) == "pending"

    def test_there_is_nothing_to_confirm_without_an_invitation(self, store):
        contacts, _ = store
        with pytest.raises(ContactError):
            contacts.confirm(AMY, MAX)

    def test_two_invitations_meeting_each_other_are_two_yes(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contact = contacts.invite(MAX, AMY)
        assert contact.confirmed, "both asked, so both agreed"

    def test_declining_removes_the_pending_row(self, store):
        contacts, conn = store
        contacts.invite(AMY, MAX)
        contacts.decline(MAX, AMY)
        assert contacts.get(AMY, MAX) is None
        assert not list(conn.execute("SELECT 1 FROM contacts"))

    def test_the_requester_cannot_decline_their_own_invitation(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        with pytest.raises(ContactError):
            contacts.decline(AMY, MAX)


class TestGate:
    def test_strangers_are_refused(self, store):
        contacts, _ = store
        with pytest.raises(ContactError):
            contacts.contact_gate(AMY, MAX)
        assert contacts.allowed(AMY, MAX) is False

    def test_a_one_sided_yes_is_not_consent(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        with pytest.raises(ContactError):
            contacts.contact_gate(AMY, MAX)
        assert contacts.allowed(AMY, MAX) is False

    def test_a_confirmed_pair_passes_the_gate_both_ways(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        assert contacts.allowed(AMY, MAX) and contacts.allowed(MAX, AMY)
        with pytest.raises(ContactError):
            contacts.contact_gate(AMY, AMY)

    def test_revoking_a_confirmed_contact_closes_the_gate(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contacts.revoke(MAX, AMY)
        assert contacts.get(AMY, MAX) is None
        assert not contacts.allowed(AMY, MAX)


class TestBlocking:
    def test_a_block_cancels_a_pending_invitation(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contact = contacts.block(MAX, AMY)
        assert contact.status == "blocked" and contact.blocked_by == MAX
        assert not contact.confirmed
        with pytest.raises(ContactError):
            contacts.contact_gate(AMY, MAX)

    def test_a_block_undoes_an_accepted_contact(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contact = contacts.block(AMY, MAX)
        assert contact.status == "blocked" and contact.confirmed_at is None
        assert contacts.allowed(AMY, MAX) is False

    def test_a_blocked_pair_cannot_be_invited_again(self, store):
        contacts, _ = store
        contacts.block(MAX, AMY)
        with pytest.raises(ContactError):
            contacts.invite(AMY, MAX)
        with pytest.raises(ContactError):
            contacts.confirm(AMY, MAX)

    def test_only_the_blocker_can_unblock(self, store):
        contacts, _ = store
        contacts.block(MAX, AMY)
        with pytest.raises(ContactError):
            contacts.unblock(AMY, MAX)
        contacts.unblock(MAX, AMY)
        assert contacts.get(AMY, MAX) is None
        assert contacts.invite(AMY, MAX).status == "pending"

    def test_unblocking_something_that_is_not_blocked_is_refused(self, store):
        contacts, _ = store
        with pytest.raises(ContactError):
            contacts.unblock(AMY, MAX)


class TestPresenceSharing:
    def test_presence_is_off_until_the_person_says_yes(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        assert contacts.presence_shared(AMY, MAX) is False
        assert contacts.presence_shared(MAX, AMY) is False

    def test_each_person_owns_their_own_flag(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contact = contacts.set_share_presence(AMY, MAX, True)
        assert contact.share_presence_a is True and contact.share_presence_b is False
        assert contacts.presence_shared(AMY, MAX) is True
        assert contacts.presence_shared(MAX, AMY) is False, "Amy's yes is not Max's"

    def test_presence_sharing_needs_a_confirmed_contact(self, store):
        contacts, _ = store
        with pytest.raises(ContactError):
            contacts.set_share_presence(AMY, MAX, True)
        contacts.invite(AMY, MAX)
        with pytest.raises(ContactError):
            contacts.set_share_presence(AMY, MAX, True)

    def test_a_block_closes_presence_too(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contacts.set_share_presence(AMY, MAX, True)
        contacts.set_share_presence(MAX, AMY, True)
        contacts.block(MAX, AMY)
        assert contacts.presence_shared(AMY, MAX) is False
        assert contacts.presence_shared(MAX, AMY) is False


class TestListing:
    def test_contacts_are_listed_for_both_sides_only(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contacts.invite(AMY, KAI)
        assert {c.other(AMY) for c in contacts.contacts_for(AMY)} == {MAX, KAI}
        assert [c.other(MAX) for c in contacts.contacts_for(MAX)] == [AMY]
        assert contacts.accepted(MAX)[0].other(MAX) == AMY
        assert contacts.accepted(KAI) == []
        assert contacts.contacts_for("") == []

    def test_another_pair_is_untouched(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contacts.invite(MAX, KAI)
        contacts.confirm(KAI, MAX)
        contacts.block(AMY, MAX)
        assert contacts.allowed(MAX, KAI) is True
        assert contacts.status(MAX, KAI) == "accepted"

    def test_a_person_needs_an_invitation_before_being_unblocked(self, store):
        contacts, _ = store
        contacts.invite(AMY, MAX)
        contacts.confirm(MAX, AMY)
        contacts.block(AMY, MAX, now=datetime.now(UTC) + timedelta(seconds=1))
        assert contacts.contacts_for(AMY)[0].status == "blocked"
