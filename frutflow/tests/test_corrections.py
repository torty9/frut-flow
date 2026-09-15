"""Accuracy regressions: intended repairs AND already-correct text.

No microphone, model, personal vocabulary, or macOS permissions required.
"""
from __future__ import annotations

import io
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flow


class PersonalizationTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name, value in {
            "CONFIG_DIR": root,
            "CORRECTIONS_PATH": root / "corrections.json",
            "CORRECTION_EXAMPLES_PATH": root / "correction_examples.json",
            "PENDING_CORRECTIONS_PATH": root / "pending_corrections.json",
            "VOCAB_PATH": root / "vocab.json",
            "_ENGLISH_WORDS": frozenset("send the message make fruit we use going server connection".split()),
        }.items():
            patch = mock.patch.object(flow, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def teach(self, heard, correct):
        flow.add_correction(heard, correct, silent=True)


class ExactCorrectionTests(PersonalizationTestCase):
    def test_longest_phrase_wins_regardless_of_teaching_order(self):
        for rules in ({"new": "fresh", "new york": "NYC"},
                      {"new york": "NYC", "new": "fresh"}):
            with self.subTest(rules=rules):
                self.assertEqual(flow.apply_corrections("new york is new", corrections=rules),
                                 "NYC is fresh")

    def test_replacements_do_not_cascade(self):
        self.assertEqual(flow.apply_corrections("Versal Vercel", corrections={
            "Versal": "Vercel", "Vercel": "Platform"}), "Vercel Platform")

    def test_cycle_applied_once(self):
        self.assertEqual(flow.apply_corrections("alpha beta", corrections={
            "alpha": "beta", "beta": "alpha"}), "beta alpha")

    def test_punctuation_ended_names_and_word_boundaries(self):
        self.assertEqual(flow.apply_corrections("C++ .NET C++17 x.NET", corrections={
            "C++": "C plus plus", ".NET": "dotnet"}),
            "C plus plus dotnet C++17 x.NET")

    def test_newer_case_variant_wins(self):
        self.assertEqual(flow.apply_corrections("FRUT frut", corrections={
            "frut": "fruit", "Frut": "früt"}), "früt früt")

    def test_unicode_case_matching_does_not_crash(self):
        self.assertEqual(flow.apply_corrections("İ", corrections={"i": "eye"}), "eye")

    def test_exact_correction_wins_over_fuzzy_repair(self):
        self.teach("frut flow", "frut Flow")
        self.teach("brand", "früt")
        cfg = dict(flow.DEFAULT_CONFIG, cleanup="none")
        self.assertEqual(flow.clean("frut flow", cfg), "frut Flow")

    def test_explicit_target_keeps_user_chosen_case(self):
        self.teach("Iphone", "iPhone")
        self.assertEqual(flow.clean("Iphone is ready", dict(flow.DEFAULT_CONFIG)),
                         "iPhone is ready")

    def test_exact_rule_inside_code_does_not_unprotect_adjacent_words(self):
        self.teach("foo", "bar")
        self.teach("brand", "Vercel")
        self.assertEqual(flow.clean("`foo Vercell`", dict(flow.DEFAULT_CONFIG, cleanup="none")),
                         "`bar Vercell`")

    def test_existing_multiword_canonical_spelling_is_protected(self):
        self.teach("brand", "Vercell Hosting")
        self.teach("host", "Vercel")
        self.assertEqual(flow.clean("Use Vercell Hosting", dict(flow.DEFAULT_CONFIG)),
                         "Use Vercell Hosting")

    def test_spanish_explicit_teaching_works(self):
        self.teach("Maria", "María")
        self.assertEqual(flow.clean("habla con Maria", dict(flow.DEFAULT_CONFIG, language="es")),
                         "Habla con María")

    def test_vocabulary_limit_also_applies_to_taught_targets(self):
        with mock.patch.object(flow, "load_corrections", return_value={
                str(i): f"Name{i}" for i in range(80)}):
            self.assertEqual(len(flow.distinctive_terms(max_terms=8)), 8)


@unittest.skipUnless(flow.FUZZY_AVAILABLE, "Install rapidfuzz and jellyfish for phonetic tests")
class FuzzyAccuracyTests(PersonalizationTestCase):
    def test_known_name_near_miss_is_repaired(self):
        self.assertEqual(flow.fuzzy_correct_text("Deploy to Versal", ["Vercel"]),
                         "Deploy to Vercel")
        self.assertEqual(flow.fuzzy_correct_text("Deploy to Vercell", ["Vercel"]),
                         "Deploy to Vercel")

    def test_canonical_accent_and_brand_case(self):
        self.assertEqual(flow.fuzzy_correct_text("Frut is ready", ["früt"]),
                         "früt is ready")

    def test_real_words_are_never_fuzzy_corrected(self):
        self.assertEqual(flow.fuzzy_correct_text("Send the message", ["Zend"]),
                         "Send the message")
        self.assertEqual(flow.fuzzy_correct_text("fruit", ["früt"]), "fruit")

    def test_competing_names_abstain_independent_of_order(self):
        for terms in (["Vercel", "Versel"], ["Versel", "Vercel"]):
            self.assertEqual(flow.fuzzy_correct_text("Versal", terms), "Versal")

    def test_spelling_alone_is_not_enough(self):
        self.assertEqual(flow.fuzzy_correct_text("Marin", ["Martin"]), "Marin")

    def test_urls_emails_paths_code_and_identifiers_unchanged(self):
        for text in ("https://Versal.com", "www.Versal.com", "hi@Versal.com",
                     "/Users/Versal/config", r"C:\Versal\file", "`Versal`",
                     "Versal_api", "Versal-123", "Versal.config", "```\nVersal\n```"):
            with self.subTest(text=text):
                self.assertEqual(flow.fuzzy_correct_text(text, ["Vercel"]), text)

    def test_short_tokens_acronyms_and_existing_spellings_unchanged(self):
        for text in ("API", "app", "FRUT", "Vercel"):
            self.assertEqual(flow.fuzzy_correct_text(text, ["früt", "Vercel", "Apple"]), text)

    def test_multiword_target_does_not_replace_one_word(self):
        self.assertEqual(flow.fuzzy_correct_text("Newark", ["New York"]), "Newark")

    def test_spanish_does_not_use_english_phonetic_matching(self):
        self.assertEqual(flow.fuzzy_correct_text("Versal", ["Vercel"], language="es"), "Versal")

    def test_high_threshold_can_disable_near_miss(self):
        self.assertEqual(flow.fuzzy_correct_text("Versal", ["Vercel"], 1.20), "Versal")

    def test_missing_dictionary_or_optional_libraries_fail_closed(self):
        with mock.patch.object(flow, "_ENGLISH_WORDS", frozenset()):
            self.assertEqual(flow.fuzzy_correct_text("Versal", ["Vercel"]), "Versal")
        with mock.patch.object(flow, "FUZZY_AVAILABLE", False):
            self.assertEqual(flow.fuzzy_correct_text("Versal", ["Vercel"]), "Versal")

    def test_empty_phonetic_codes_are_not_evidence(self):
        encoder = mock.Mock()
        encoder.metaphone.return_value = ""
        encoder.soundex.return_value = ""
        with mock.patch.object(flow, "_jellyfish", encoder):
            self.assertFalse(flow._phonetic_match("aaa", "eee"))
        self.assertFalse(flow._phonetic_match("東京", "京都"))


class ModelRepairValidationTests(PersonalizationTestCase):
    def test_valid_homophone_repairs(self):
        for src, out in (
            ("Your going to love this feature", "You're going to love this feature"),
            ("The server lost it's connection", "The server lost its connection"),
            ("Put the boxes over they're by the door", "Put the boxes over there by the door"),
            ("Meet at the peer before the boat leaves", "Meet at the pier before the boat leaves"),
            ("We should of tested it first", "We should have tested it first"),
            ("I want to by two new monitors", "I want to buy two new monitors"),
            ("Its to cold to go outside", "It's too cold to go outside"),
        ):
            with self.subTest(src=src):
                self.assertEqual(flow._validated_repair(src, out), out)

    def test_numbers_negation_deletions_and_insertions_rejected(self):
        for src, out in (
            ("Send 15 files today", "Send 50 files today"),
            ("Do not send it", "Do send it"),
            ("We can ship it tomorrow", "We can't ship it tomorrow"),
            ("We can't ship it tomorrow", "We can ship it tomorrow"),
            ("Ship tomorrow", "Ship it tomorrow"),
            ("Send it", "Delete it"),
            ("I ordered the blue version yesterday", "I ordered the new version yesterday"),
            ("The release is ready for the team", "The release is ready"),
        ):
            with self.subTest(src=src):
                self.assertIsNone(flow._validated_repair(src, out))

    def test_literals_and_taught_names_are_protected(self):
        for src, out in (
            ("Go to https://Versal.com today", "Go to https://Vercel.com today"),
            ("The `peer` field is empty", "The `pier` field is empty"),
            ("Send to peer@example.com today", "Send to pier@example.com today"),
            ("Please use peer_id here", "Please use pier_id here"),
        ):
            self.assertIsNone(flow._validated_repair(src, out))
        self.assertIsNone(flow._validated_repair(
            "Work at Versal today", "Work at Vercel today", known_terms=["Versal"]))

    def test_untaught_distinctive_name_cannot_be_rewritten_by_model(self):
        self.assertIsNone(flow._validated_repair("We hired Vercell today", "We hired Vercel today"))

    def test_all_taught_targets_protected_beyond_glossary_limit(self):
        with mock.patch.object(flow, "load_corrections", return_value={
                **{f"heard{i}": f"Brand{i}" for i in range(65)}, "dock": "peer"}):
            self.assertEqual(self.repair("Meet at the peer before noon", "Meet at the pier before noon"),
                             "Meet at the peer before noon")

    def test_existing_format_and_case_preserved(self):
        self.assertEqual(flow._validated_repair("Use the API today.", "use the api today."),
                         "Use the API today.")
        self.assertIsNone(flow._validated_repair("First\n\nsecond", "First second"))
        self.assertIsNone(flow._validated_repair("Ready?", "Ready."))

    def test_same_source_quotes_are_kept(self):
        self.assertEqual(self.repair('"Ready for launch"', '"Ready for launch"'),
                         '"Ready for launch"')

    def repair(self, src, proposal):
        rep = mock.Mock()
        rep.generate.return_value = proposal
        with mock.patch.object(flow, "_get_local_repairer", return_value=rep):
            return flow.local_repair(src, dict(flow.DEFAULT_CONFIG, cleanup="local"))

    def test_added_wrapper_and_terminal_period_are_removed(self):
        self.assertEqual(self.repair("Your going to love this feature",
                                     '"You\'re going to love this feature."'),
                         "You're going to love this feature")

    def test_bad_model_output_falls_back_to_source(self):
        self.assertEqual(self.repair("Do not send it", "Do send it"), "Do not send it")

    def test_model_failure_preserves_explicit_correction(self):
        self.teach("Versal", "Vercel")
        with mock.patch.object(flow, "local_repair", side_effect=RuntimeError("offline")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(flow.clean("Deploy to Versal", dict(flow.DEFAULT_CONFIG, cleanup="local")),
                             "Deploy to Vercel")

    def test_only_relevant_examples_enter_prompt(self):
        self.teach("Versal", "Vercel")
        self.teach("Frut", "früt")
        examples = flow._relevant_correction_examples({"app": "Editor"}, text="Deploy to Vercel")
        self.assertEqual([e["correct"] for e in examples], ["Vercel"])

    def test_unknown_short_or_non_english_output_fails_closed(self):
        self.assertIsNone(flow._validated_repair("your ready", "you're ready"))
        self.assertIsNone(flow._validated_repair("Envía el archivo mañana", "Send the file tomorrow", language="es"))


class EditLearningTests(PersonalizationTestCase):
    def learn(self, before="Deploy to Versal now", after="Deploy to Vercel now", *, baseline=None):
        with mock.patch.object(flow, "_ax_read_value", return_value=after), \
                redirect_stdout(io.StringIO()):
            return flow.learn_from_edit(object(), before,
                                        baseline=before if baseline is None else baseline,
                                        app="Editor")

    def test_two_separate_edits_required_then_context_is_saved(self):
        self.assertEqual(self.learn(), 0)
        self.assertEqual(flow.load_corrections(), {})
        self.assertEqual(flow.distinctive_terms(), [])
        self.assertEqual(flow.load_correction_examples(), {})
        self.assertEqual(self.learn(), 1)
        self.assertEqual(flow.load_corrections(), {"Versal": "Vercel"})
        self.assertEqual(flow.load_correction_examples()["Versal"]["app"], "Editor")

    def test_repeated_words_in_one_paste_count_once(self):
        before = "Deploy to Versal now and check Versal again today"
        after = "Deploy to Vercel now and check Vercel again today"
        self.assertEqual(self.learn(before, after), 0)
        self.assertEqual(flow.load_corrections(), {})

    def test_edits_outside_inserted_text_are_ignored(self):
        before = "Header Versal\nDeploy to Versal now\nFooter"
        after = "Header Vercel\nDeploy to Versal now\nFooter"
        for _ in range(2):
            self.assertEqual(self.learn(after=after, baseline=before), 0)
        self.assertEqual(flow.load_corrections(), {})

    def test_inserted_text_can_be_corrected_inside_document(self):
        before = "Header\nDeploy to Versal now\nFooter"
        after = "Header\nDeploy to Vercel now\nFooter"
        self.assertEqual(self.learn(after=after, baseline=before), 0)
        self.assertEqual(self.learn(after=after, baseline=before), 1)

    def test_missing_or_ambiguous_baseline_never_learns(self):
        self.assertIsNone(flow._edited_dictation("Vercel", "Versal", None))
        self.assertIsNone(flow._edited_dictation("Vercel Versal", "Versal", "Versal Versal"))
        self.assertIsNone(flow._edited_dictation("a" * 17000, "Versal", "Versal"))

    def test_manual_rule_cannot_be_overwritten_by_auto_learning(self):
        self.teach("Versal", "MyBrand")
        self.learn()
        self.learn()
        self.assertEqual(flow.load_corrections()["Versal"], "MyBrand")

    def test_sentence_rewrites_and_technical_edits_not_learned(self):
        for before, after in (
            ("We can ship to Versal today", "We can't ship to Vercel today"),
            ("Use Versal tomorrow", "Use Vercel next Monday"),
            ("Open https://Versal.com now", "Open https://Vercel.com now"),
            ("Send 15 to Versal now", "Send 50 to Vercel now"),
            ("Send it to Maria", "Send it to Martin"),
        ):
            with self.subTest(before=before):
                self.assertEqual(self.learn(before, after), 0)
                self.assertEqual(self.learn(before, after), 0)
        self.assertEqual(flow.load_corrections(), {})

    def test_conflicting_observation_resets_confirmation(self):
        self.assertFalse(flow._observe_correction("Versal", "Vercel", context="", app="Editor"))
        self.assertFalse(flow._observe_correction("Versal", "Versel", context="", app="Editor"))
        self.assertFalse(flow._observe_correction("Versal", "Vercel", context="", app="Editor"))
        self.assertEqual(flow.load_corrections(), {})

    def test_timer_captures_snapshot_and_only_reconciles_once(self):
        app = flow.FlowApp.__new__(flow.FlowApp)
        app.cfg = {}
        app._state_lock = threading.RLock()
        app._learn_timer = None
        app._learn_generation = 0
        app._pending_learn = None
        with mock.patch.object(flow, "_ax_focused_element", return_value="element"), \
                mock.patch.object(flow, "_ax_read_value", return_value="Header Deploy to Versal now"), \
                mock.patch.object(flow, "_focused_app_name", return_value="Editor"), \
                mock.patch.object(flow.threading, "Timer"), \
                mock.patch.object(flow, "learn_from_edit") as learn:
            app._arm_edit_learning("Deploy to Versal now")
            generation = app._learn_generation
            app._reconcile_edit_learning(generation - 1)
            learn.assert_not_called()
            app._reconcile_edit_learning(generation)
            app._reconcile_edit_learning(generation)
            learn.assert_called_once_with("element", "Deploy to Versal now",
                baseline="Header Deploy to Versal now", app="Editor")


if __name__ == "__main__":
    unittest.main()
