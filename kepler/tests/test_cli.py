"""Tests for kepler.cli."""


import os
from shutil import rmtree
from subprocess import check_output, STDOUT, check_call
from tempfile import mkdtemp

import pandas as pd

from kepler.tests import TestKepler
from kepler.utils import get_engine


class TestCLI(TestKepler):

    def test_setup(self):
        """Check if kepler initializes in the right place properly."""
        cfgdir = mkdtemp()
        cmd = f'kepler setup --path {cfgdir}'.split()
        try:
            out = check_output(cmd)
            self.assertTrue(out.decode('utf-8').rstrip().endswith('Welcome to Kepler!'))
            self.check_config(cfgdir)
        finally:
            rmtree(cfgdir)

    def test_add_project(self):
        """Test if the add project command works."""
        # nonexistent folder
        os.environ['KEPLER_HOME'] = self.config_dir
        cmd = 'kepler add project -n foo -p /tmp/foo'.split()
        out = check_output(cmd, stderr=STDOUT)
        self.assertTrue(out.rstrip().decode('utf8').endswith(
            'No such directory: /tmp/foo'))

        cmd = 'kepler add project -n foo -p .'.split()
        check_call(cmd)
        engine = get_engine()
        df = pd.read_sql_table('projects', engine)
        p = df[df['name'] == 'foo'].iloc[0]
        self.assertEqual(p['location'], os.getcwd())
