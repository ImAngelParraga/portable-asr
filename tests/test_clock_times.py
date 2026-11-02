import unittest
from unittest.mock import patch

import asr_server


class ClockTimesTest(unittest.TestCase):
    def clean(self, raw, output):
        with (
            patch.object(asr_server, 'ASR_POSTPROCESS_ENABLED', True),
            patch.object(asr_server, 'ASR_POSTPROCESS_VALIDATE_EDITS', True),
            patch.object(asr_server, '_llm_chat', return_value={
                'choices': [{'message': {'content': output}}], 'usage': {},
            }) as llm,
        ):
            result = asr_server._postprocess_transcript(raw)
            self.assertEqual(llm.call_count, 1)
            return result

    def test_v52_reported_times_with_unchanged_or_formatted_model_output(self):
        raw = ('Vale, pero llego un poco justo. El oficio es a las seis y media, '
               'salgo a las siete y media y a casa llegaré a las ocho menos cuarto. '
               'Sería cambiar el coche y salir corriendo para ir al tenis.')
        expected = raw.replace('seis y media', '6:30').replace('siete y media', '7:30').replace('ocho menos cuarto', '7:45')
        for output in (raw, expected):
            with self.subTest(output=output):
                self.assertEqual(self.clean(raw, output), expected)

    def test_v52_clock_forms_and_hour_rollover(self):
        for raw, expected in (
            ('Son las tres y cuarto.', 'Son las 3:15.'),
            ('Es la una menos cuarto.', 'Es la 12:45.'),
            ('A las doce menos cinco.', 'A las 11:55.'),
            ('Desde las veintitrés y diez hasta las cero y cinco.', 'Desde las 23:10 hasta las 0:05.'),
            ('Sobre las nueve en punto.', 'Sobre las 9:00.'),
            ('A la una.', 'A la 1:00.'),
            ('A las seis y veinticinco de la tarde.', 'A las 6:25 de la tarde.'),
            ('A las 6 y media.', 'A las 6:30.'),
            ('A las 06:30.', 'A las 06:30.'),
            ('A las 5.', 'A las 5.'),
            ('A las siete y treinta y cinco.', 'A las 7:35.'),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(self.clean(raw, raw), expected)

    def test_v52_changed_times_and_surrounding_words_are_rejected(self):
        raw = 'Llego a las ocho menos cuarto mañana.'
        for output in ('Llego a las 8:45 mañana.', 'Llego a las 7:30 mañana.',
                       'Llego a las 7:45 hoy.', 'Llego a las 19:45 mañana.'):
            with self.subTest(output=output):
                self.assertEqual(self.clean(raw, output), 'Llego a las 7:45 mañana.')

    def test_v52_ambiguous_invalid_and_duration_phrases_remain_literal(self):
        for raw in ('Seis y media.', 'Trabajé seis horas y media.', 'Compré seis y media docenas.',
                    'A las seis y sesenta.', 'A las veinticinco y media.',
                    'A las seis menos setenta.', 'A las dos horas de salir.',
                    'Las seis y media son mis favoritas.', 'A las seis y pico.'):
            with self.subTest(raw=raw):
                self.assertEqual(self.clean(raw, raw), raw)

    def test_v52_questions_and_lists_keep_clock_formatting(self):
        self.assertEqual(self.clean('pregunta llegas a las seis y media',
                                    'Llegas a las seis y media.'),
                         '¿Llegas a las 6:30?')
        raw = 'agenda lista de ítems salir a las seis y media nuevo ítem volver a las ocho menos cuarto'
        expected = 'agenda:\n\n- salir a las 6:30\n- volver a las 7:45'
        self.assertEqual(self.clean(raw, expected), expected)

    def test_v52_disabled_and_unavailable_cleanup_keep_raw_text(self):
        raw = 'Llego a las seis y media.'
        with patch.object(asr_server, 'ASR_POSTPROCESS_ENABLED', False):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
        with (patch.object(asr_server, 'ASR_POSTPROCESS_ENABLED', True),
              patch.object(asr_server, '_llm_chat', side_effect=RuntimeError('unavailable'))):
            self.assertEqual(asr_server._postprocess_transcript(raw), raw)
