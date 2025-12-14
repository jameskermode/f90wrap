import unittest
from f90wrap import fortran, parser, transform
from . import test_samples_dir


class TestTransform(unittest.TestCase):

    def setUp(self):
        self.root = parser.read_files([str(test_samples_dir/'circle.f90')])

    def test_resolve_interface_prototypes(self):
        ''' Verify procedures gets moved into interface objects '''
        new = transform.ResolveInterfacePrototypes().visit(self.root)
        m = new.modules[0]
        self.assertEqual(len(m.procedures), 6)
        self.assertTrue(isinstance(
            m.interfaces[0].procedures[0],
            fortran.Function
        ))

    def test_parse_dnad(self):
        root = parser.read_files([str(test_samples_dir/'DNAD.fpp')])
        new = transform.ResolveInterfacePrototypes().visit(root)
        m = new.modules[0]
        self.assertEqual(len(m.procedures), 1)
        # TODO: Fix incomplete resolution of prototypes
        #       This is because both interfaces reference the same procedure
        #       but we only resolve first reference of a given procedure.
        self.assertIsInstance(m.interfaces[12].procedures[0], fortran.Function)
        self.assertIsInstance(m.interfaces[13].procedures[0], fortran.Prototype)

    def test_resolve_binding_prototypes(self):
        ''' Verify procedures gets moved into binding objects '''
        new = transform.ResolveBindingPrototypes().visit(self.root)
        m = new.modules[0]
        t = m.types[0]
        b_normal = t.bindings[0]
        b_generic = t.bindings[2]
        b_final = t.bindings[3]
        self.assertEqual(len(m.procedures), 2)
        self.assertEqual(len(b_normal.procedures), 1)
        self.assertEqual(len(b_generic.procedures), 2)
        self.assertIn('destructor', b_final.attributes)
        self.assertTrue(isinstance(
            b_normal.procedures[0],
            fortran.Function
        ))

    def test_bind_constructor_interfaces(self):
        ''' Verify interfaces with same name as type become constructors '''
        new = transform.ResolveInterfacePrototypes().visit(self.root)
        new = transform.BindConstructorInterfaces().visit(new)
        m = new.modules[0]
        t = m.types[0]
        self.assertEqual(len(m.interfaces), 0)
        self.assertEqual(len(t.interfaces), 1)
        self.assertIn('constructor', t.interfaces[0].attributes)

    def test_generic_tranform(self):
        types = fortran.find_types(self.root)
        mods  = { type.mod_name: type.mod_name for _,type in types.items()}
        new = transform.transform_to_generic_wrapper(self.root,
            types=types,
            callbacks=[],
            constructors=[],
            destructors=[],
            short_names={},
            init_lines={},
            kept_subs=[],
            kept_mods=[],
            argument_name_map={},
            move_methods=True,
            shorten_routine_names=[],
            modules_for_type=mods,
            remove_optional_arguments=False,
            force_public=[],
        )
        m = new.modules[0]
        t = m.types[0]
        self.assertEqual(len(m.procedures), 0)
        self.assertEqual(len(t.elements), 0)
        self.assertEqual(len(t.bindings), 4)
        self.assertEqual(len(t.interfaces), 1)

    def test_shorten_long_name(self):
        '''
        Verify that shorten_long_name correctly truncates names exceeding 63 characters.
        This is a regression test for issue #120.
        '''
        from f90wrap.transform import shorten_long_name

        # Short name should pass through unchanged
        short_name = "short_name"
        self.assertEqual(shorten_long_name(short_name), short_name)

        # Name exactly 63 chars should pass through unchanged
        exact_name = "a" * 63
        self.assertEqual(shorten_long_name(exact_name), exact_name)

        # Long name should be shortened to 63 chars with hash suffix
        long_name = "this_is_a_very_long_variable_name_that_exceeds_the_sixty_three_character_limit"
        shortened = shorten_long_name(long_name)
        self.assertEqual(len(shortened), 63)
        # Should start with truncated prefix and end with hash
        self.assertTrue(shortened.startswith("this_is_a_very_long_variable_name_that_exceeds_the_sixty_th"))

        # Same input should produce same output (deterministic)
        self.assertEqual(shorten_long_name(long_name), shortened)

    def test_kind_parameter_uses_clause(self):
        '''
        Verify that kind parameters used in procedure arguments are imported.
        This is a regression test for issue #253.
        '''
        root = parser.read_files([str(test_samples_dir/'kind_param.f90')])
        types = fortran.find_types(root)

        # Apply the transformation that sets up uses clauses
        transform.fix_subroutine_uses_clauses(root, types)

        # Find the multiply function
        m = root.modules[0]
        func = m.procedures[0]
        self.assertEqual(func.name, 'multiply')

        # Verify that jprb is in the uses clause
        uses_symbols = set()
        for mod_name, symbols in func.uses:
            for sym in symbols:
                uses_symbols.add(sym)

        self.assertIn('jprb', uses_symbols,
            'Kind parameter jprb should be included in uses clause')
