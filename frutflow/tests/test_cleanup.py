"""Tests for the language-aware cleanup pipeline: spoken punctuation (quotes!),
Spanish support, ASR hallucination filtering, and auto language detection.

Pure text-level tests — no models, no audio, no AppKit.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flow  # noqa: E402  imported after adding the repository root


def en(text: str) -> str:
    return flow.basic_cleanup(text, language="en")


def es(text: str) -> str:
    return flow.basic_cleanup(text, language="es")


class QuoteCommandTests(unittest.TestCase):
    def test_bare_quote_pairs_with_explicit_closer(self):
        self.assertEqual(en("she said quote I'll be there end quote"),
                         'She said "I\'ll be there"')

    def test_engine_commas_around_commands_are_absorbed(self):
        self.assertEqual(en("She said, quote, I'll be there, end quote."),
                         'She said, "I\'ll be there".')

    def test_unquote_closer(self):
        self.assertEqual(en("the sign says quote no entry unquote"),
                         'The sign says "no entry"')

    def test_explicit_open_close(self):
        self.assertEqual(en("open quote hello close quote"), '"Hello"')

    def test_unclosed_explicit_opener_quotes_rest_of_utterance(self):
        self.assertEqual(en("tell her open quote I'll be late"),
                         'Tell her "I\'ll be late"')

    def test_bare_quote_without_closer_is_left_alone(self):
        self.assertEqual(en("can you get a quote from the plumber"),
                         "Can you get a quote from the plumber")

    def test_quote_unquote_wraps_next_word(self):
        self.assertEqual(en("the quote unquote expert"), 'The "expert"')

    def test_quotation_mark_pair(self):
        self.assertEqual(en("quotation mark done quotation mark"), '"Done"')

    def test_spanish_abrir_cerrar_comillas(self):
        self.assertEqual(es("abrir comillas hola cerrar comillas"), '"Hola"')

    def test_spanish_apple_latam_forms(self):
        self.assertEqual(
            es("comillas de apertura buenos días comillas de cierre"),
            '"Buenos días"')

    def test_spanish_comillas_dobles(self):
        self.assertEqual(es("abrir comillas dobles ya voy cerrar comillas dobles"),
                         '"Ya voy"')

    def test_spanish_entre_comillas_paired(self):
        self.assertEqual(es("es entre comillas urgente cerrar comillas"),
                         'Es "urgente"')

    def test_spanish_mentioning_comillas_is_untouched(self):
        self.assertEqual(es("no sé dónde van las comillas en esta frase"),
                         "No sé dónde van las comillas en esta frase")

    def test_double_period_around_closing_quote_spanish_rae_style(self):
        # The engine often ends BOTH the quoted span and the sentence with "."
        # — Spanish keeps the mark outside the quote (RAE).
        self.assertEqual(
            es("Dile abre comillas, ya voy en camino. Cierra comillas. Gracias."),
            'Dile "ya voy en camino". Gracias.')

    def test_double_period_around_closing_quote_english_style(self):
        # English keeps the period inside the closing quote.
        self.assertEqual(en("she said quote on my way. end quote. thanks"),
                         'She said "on my way." thanks')


class SpokenPunctuationEnglishTests(unittest.TestCase):
    def test_new_paragraph_capitalizes(self):
        self.assertEqual(en("first point new paragraph second point"),
                         "First point\n\nSecond point")

    def test_new_line(self):
        # "new line" breaks the line but (like Apple) does not force a capital
        self.assertEqual(en("first line new line second line"),
                         "First line\nsecond line")

    def test_question_mark(self):
        self.assertEqual(en("are you coming question mark"), "Are you coming?")

    def test_question_mark_absorbs_duplicate(self):
        self.assertEqual(en("are you coming question mark?"), "Are you coming?")

    def test_exclamation_point(self):
        self.assertEqual(en("stop right there exclamation point"),
                         "Stop right there!")

    def test_period_at_end(self):
        self.assertEqual(en("send it now period"), "Send it now.")

    def test_period_mid_sentence_capitalizes_next(self):
        self.assertEqual(en("it works period and that's final"),
                         "It works. And that's final")

    def test_period_word_collapse_with_engine_period(self):
        self.assertEqual(en("it works period. next steps"),
                         "It works. Next steps")

    def test_trial_period_is_prose(self):
        self.assertEqual(en("the trial period ends tomorrow"),
                         "The trial period ends tomorrow")

    def test_period_of_time_is_prose(self):
        self.assertEqual(en("a long period of silence followed"),
                         "A long period of silence followed")

    def test_comma_list(self):
        self.assertEqual(en("apples comma bananas comma pears"),
                         "Apples, bananas, pears")

    def test_oxford_comma_is_prose(self):
        self.assertEqual(en("use the oxford comma in this style guide"),
                         "Use the oxford comma in this style guide")

    def test_colon_medical_is_prose(self):
        self.assertEqual(en("colon cancer screening is important"),
                         "Colon cancer screening is important")

    def test_colon_command(self):
        self.assertEqual(en("here's the plan colon ship it"),
                         "Here's the plan: ship it")

    def test_hyphen_joins(self):
        self.assertEqual(en("twenty hyphen five"), "Twenty-five")

    def test_dot_com(self):
        self.assertEqual(en("go to frut dot com now"), "Go to frut.com now")

    def test_underscore_joins(self):
        self.assertEqual(en("john underscore smith"), "John_smith")

    def test_semicolon(self):
        self.assertEqual(en("do it semicolon then rest"), "Do it; then rest")


class SpokenPunctuationSpanishTests(unittest.TestCase):
    def test_punto_at_end(self):
        self.assertEqual(es("dile que sí punto"), "Dile que sí.")

    def test_punto_de_vista_is_prose(self):
        self.assertEqual(es("es un punto de vista interesante"),
                         "Es un punto de vista interesante")

    def test_hasta_cierto_punto_is_prose(self):
        self.assertEqual(es("estoy de acuerdo hasta cierto punto"),
                         "Estoy de acuerdo hasta cierto punto")

    def test_coma(self):
        self.assertEqual(es("pan coma leche coma huevos"),
                         "Pan, leche, huevos")

    def test_en_coma_is_prose(self):
        self.assertEqual(es("el paciente está en coma inducido"),
                         "El paciente está en coma inducido")

    def test_signo_de_interrogacion(self):
        self.assertEqual(es("vienes mañana signo de interrogación"),
                         "Vienes mañana?")

    def test_abrir_interrogacion(self):
        self.assertEqual(es("abrir interrogación vienes signo de interrogación"),
                         "¿Vienes?")

    def test_signo_de_admiracion(self):
        self.assertEqual(es("qué bueno signo de admiración"), "Qué bueno!")

    def test_punto_y_aparte(self):
        self.assertEqual(es("gracias punto y aparte nos vemos"),
                         "Gracias.\n\nNos vemos")

    def test_punto_y_aparte_after_engine_period(self):
        self.assertEqual(es("gracias. punto y aparte nos vemos"),
                         "Gracias.\n\nNos vemos")

    def test_nueva_linea(self):
        # "nueva línea" breaks the line but (like Apple) does not force a capital
        self.assertEqual(es("primero nueva línea segundo"), "Primero\nsegundo")

    def test_punto_y_coma(self):
        self.assertEqual(es("hazlo punto y coma luego descansa"),
                         "Hazlo; luego descansa")

    def test_dos_puntos_command(self):
        self.assertEqual(es("necesito lo siguiente dos puntos pan y leche"),
                         "Necesito lo siguiente: pan y leche")

    def test_dos_puntos_score_is_prose(self):
        self.assertEqual(es("ganamos por dos puntos"), "Ganamos por dos puntos")

    def test_arroba_and_punto_com(self):
        self.assertEqual(es("escribe a maría arroba gmail punto com"),
                         "Escribe a maría@gmail.com")

    def test_guion_bajo(self):
        self.assertEqual(es("usuario guion bajo nuevo"), "Usuario_nuevo")

    def test_capitalizes_after_inverted_question_mark(self):
        self.assertEqual(es("¿hola?"), "¿Hola?")

    def test_no_space_inside_inverted_marks(self):
        self.assertEqual(es("¿ cómo estás ?"), "¿Cómo estás?")


class FillerTests(unittest.TestCase):
    def test_english_fillers_stripped(self):
        self.assertEqual(en("um hello there"), "Hello there")

    def test_spanish_ehm_stripped(self):
        self.assertEqual(es("ehm quiero dos"), "Quiero dos")

    def test_spanish_eh_is_kept(self):
        # "eh" is a real Spanish interjection — must never be stripped.
        self.assertEqual(es("eh tú ven aquí"), "Eh tú ven aquí")

    def test_spanish_este_is_kept(self):
        self.assertEqual(es("este libro es bueno"), "Este libro es bueno")


class HallucinationTests(unittest.TestCase):
    def _clean(self, text, mode="basic", language="en"):
        cfg = dict(flow.DEFAULT_CONFIG)
        cfg["cleanup"] = mode
        cfg["language"] = language
        cfg["fuzzy_correct"] = False
        return flow.clean(text, cfg)

    def test_amara_spanish_dropped(self):
        self.assertEqual(
            self._clean("Subtítulos realizados por la comunidad de Amara.org",
                        language="es"), "")

    def test_thanks_for_watching_dropped(self):
        self.assertEqual(self._clean("Thanks for watching!"), "")

    def test_dropped_even_in_none_mode(self):
        self.assertEqual(self._clean("Thanks for watching!", mode="none"), "")

    def test_music_tag_dropped(self):
        self.assertEqual(self._clean("[Música]", language="es"), "")

    def test_plain_thank_you_survives(self):
        self.assertEqual(self._clean("Thank you."), "Thank you.")

    def test_plain_gracias_survives(self):
        self.assertEqual(self._clean("Gracias.", language="es"), "Gracias.")

    def test_repetition_loop_collapses(self):
        self.assertEqual(self._clean("you you you you you you"), "You")

    def test_deliberate_no_no_no_with_commas_survives(self):
        self.assertEqual(self._clean("No, no, no, that's wrong."),
                         "No, no, no, that's wrong.")


class LanguageDetectionTests(unittest.TestCase):
    def test_spanish_detected(self):
        self.assertEqual(
            flow._detect_cleanup_language("¿Dónde está el baño?"), "es")

    def test_english_detected(self):
        self.assertEqual(
            flow._detect_cleanup_language("Let's meet at the café tomorrow"),
            "en")

    def test_auto_mode_applies_spanish_rules(self):
        cfg = dict(flow.DEFAULT_CONFIG)
        cfg["cleanup"] = "basic"
        cfg["language"] = "auto"
        cfg["fuzzy_correct"] = False
        self.assertEqual(flow.clean("hola señora coma qué tal", cfg),
                         "Hola señora, qué tal")

    def test_auto_mode_applies_english_rules(self):
        cfg = dict(flow.DEFAULT_CONFIG)
        cfg["cleanup"] = "basic"
        cfg["language"] = "auto"
        cfg["fuzzy_correct"] = False
        self.assertEqual(flow.clean("send it now period", cfg), "Send it now.")


class ModelResolutionTests(unittest.TestCase):
    def test_parakeet_v2_upgraded_for_spanish(self):
        cfg = {"language": "es",
               "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v2"}
        self.assertEqual(flow._resolve_parakeet_model(cfg),
                         "mlx-community/parakeet-tdt-0.6b-v3")

    def test_parakeet_v2_kept_for_english(self):
        cfg = {"language": "en",
               "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v2"}
        self.assertEqual(flow._resolve_parakeet_model(cfg),
                         "mlx-community/parakeet-tdt-0.6b-v2")

    def test_parakeet_v3_kept_as_is(self):
        cfg = {"language": "es",
               "parakeet_model": "mlx-community/parakeet-tdt-0.6b-v3"}
        self.assertEqual(flow._resolve_parakeet_model(cfg),
                         "mlx-community/parakeet-tdt-0.6b-v3")

    def test_whisper_distil_swapped_for_spanish(self):
        cfg = {"language": "es", "model": "distil-large-v3"}
        self.assertEqual(flow._resolve_whisper_model(cfg), "large-v3-turbo")

    def test_whisper_distil_swapped_for_auto(self):
        cfg = {"language": "auto", "model": "distil-large-v3"}
        self.assertEqual(flow._resolve_whisper_model(cfg), "large-v3-turbo")

    def test_whisper_distil_kept_for_english(self):
        cfg = {"language": "en", "model": "distil-large-v3"}
        self.assertEqual(flow._resolve_whisper_model(cfg), "distil-large-v3")

    def test_whisper_multilingual_kept_for_spanish(self):
        cfg = {"language": "es", "model": "large-v3"}
        self.assertEqual(flow._resolve_whisper_model(cfg), "large-v3")


class SpanishUndoPhraseTests(unittest.TestCase):
    def test_olvidalo_retracts_previous_dictation(self):
        kept, prev_delete = flow.apply_undo("Olvídalo.", flow.DEFAULT_CONFIG)
        self.assertEqual((kept, prev_delete), ("", 1))

    def test_borra_eso_retracts_previous_dictation(self):
        kept, prev_delete = flow.apply_undo("borra eso", flow.DEFAULT_CONFIG)
        self.assertEqual((kept, prev_delete), ("", 1))


class RegressionTests(unittest.TestCase):
    def test_plain_english_unchanged(self):
        self.assertEqual(en("  hello   world , yes "), "Hello world, yes")

    def test_leading_digit_not_capitalized(self):
        self.assertEqual(en("20 people came"), "20 people came")

    def test_spanish_text_with_english_config_keeps_words(self):
        # Forced-English cleanup on Spanish words must not delete anything
        # (only the "en" filler set applies).
        self.assertEqual(en("hola qué tal todo bien"), "Hola qué tal todo bien")


if __name__ == "__main__":
    unittest.main()
