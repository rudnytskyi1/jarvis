"""Absent requested people use saved examples without inventing scene identity."""
import pytest

from hub.image_subjects import select_image_subjects

NAMES = ['Anton', 'John', 'Theodric']


def select(text, explicit=(), speaker='Anton', faces=(), source='camera', names=NAMES):
    return select_image_subjects(text, explicit, names, speaker, source=source, faces_in_frame=faces)


def test_absent_named_person_is_selected_even_when_model_omits_reference_argument():
    result = select('Make a picture where Anton is sitting next to him.', faces=[{'name': None}])
    assert result['requested_people'] == result['reference_people'] == ['Anton']
    assert result['visible_people'] == result['ambiguous_people'] == []


def test_exact_frame_match_avoids_unnecessary_saved_reference():
    result = select('Make Anton a superhero.', faces=[{'name': 'Anton'}, {'name': None}])
    assert result['visible_people'] == ['Anton'] and not result['reference_people']


def test_duplicate_name_in_exact_frame_remains_ambiguous():
    result = select('Make Anton a superhero.', faces=[{'name': 'Anton'}, {'name': 'anton'}])
    assert result['ambiguous_people'] == ['Anton']
    assert not result['reference_people'] and not result['visible_people']


@pytest.mark.parametrize('text', ['Put me beside him.', 'Make me look like a superhero.',
                                 'Put a hat on my head.', 'Нарисуй меня рядом с ним.',
                                 'Надень мне шляпу.'])
def test_depicted_first_person_uses_identified_speaker(text):
    assert select(text)['reference_people'] == ['Anton']


@pytest.mark.parametrize('text', ['Draw me a cat.', 'Can you generate me a picture of a cat?',
                                 'Make me a picture of a mountain.', 'Draw my dog.',
                                 'Нарисуй мне кота.', 'Нарисуй мой дом.'])
def test_recipient_or_unrelated_possession_does_not_upload_speaker_photo(text):
    assert not select(text, explicit=['me'])['reference_people']


def test_unknown_voice_does_not_guess_speaker_from_people_in_room():
    result = select('Put me beside him.', speaker='unknown', faces=[{'name': 'John'}])
    assert not result['requested_people']


def test_known_other_person_does_not_fill_in_unknown_speaker():
    assert not select('Make me a superhero.', speaker='', explicit=['John'])['requested_people']


def test_source_without_camera_uses_all_explicitly_requested_identities():
    result = select('Draw Anton next to Theodric.', source='none')
    assert result['reference_people'] == ['Anton', 'Theodric']


def test_mixed_visible_and_absent_people_only_require_absent_gallery():
    result = select('Place John beside me.', faces=[{'name': 'Anton'}])
    assert result['visible_people'] == ['Anton']
    assert result['reference_people'] == ['John']


@pytest.mark.parametrize('text', ['Draw a cat without Anton.', 'Do not add Anton.',
                                 'Draw Anton, but Anton should not be in it.',
                                 'Draw the words "Anton".', 'Draw a sign reading Anton.',
                                 'Draw Anton. Cancel that.'])
def test_tool_argument_cannot_upload_excluded_or_literal_caption_identity(text):
    assert not select(text, explicit=['Anton'])['reference_people']


def test_excluded_name_cannot_reappear_through_speaker_pronoun():
    assert not select('Draw me, but do not include Anton.', explicit=['me'])['reference_people']


def test_unrequested_registered_name_and_unregistered_name_are_not_added():
    result = select('Draw Anton beside a tree.', explicit=['John', 'Unregistered'])
    assert result['reference_people'] == ['Anton']


def test_long_profile_name_does_not_select_its_short_prefix_profile():
    result = select('Draw John the system beside a tree.', names=['John', 'John the system'])
    assert result['reference_people'] == ['John the system']


def test_separately_requested_short_and_long_names_are_both_retained():
    result = select('Draw John and John the system.', names=['John', 'John the system'])
    assert result['reference_people'] == ['John', 'John the system']


def test_no_silent_two_person_truncation_or_fuzzy_name_matching():
    assert select('Draw Anton, John and Theodric.', source='none')['reference_people'] == NAMES
    assert not select('Draw Anthony next to him.')['reference_people']


def test_case_and_explicit_duplicate_aliases_resolve_once():
    result = select('Draw ANTON next to me.', explicit=['anton', 'me', 'Anton'])
    assert result['reference_people'] == ['Anton']
