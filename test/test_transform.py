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
