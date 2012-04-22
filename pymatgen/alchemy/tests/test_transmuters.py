#!/usr/bin/env python

'''
Created on Mar 5, 2012
'''

from __future__ import division

__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyue@mit.edu"
__date__ = "Mar 5, 2012"

import unittest
import os
from pymatgen.alchemy.transmuters import TransformedStructureTransmuter
from pymatgen.transformations.standard_transformations import SubstitutionTransformation, RemoveSpeciesTransformation, OrderDisorderedStructureTransformation, SuperTransformation

import pymatgen
test_dir = os.path.join(os.path.dirname(os.path.abspath(pymatgen.__file__)), '..', 'test_files')


class TransformedStructureTransmuterTest(unittest.TestCase):

    def test_cif_transmute(self):
        trans = []
        trans.append(SubstitutionTransformation({"Fe":"Mn", "Fe2+":"Mn2+"}))
        tsc = TransformedStructureTransmuter.from_cifs([os.path.join(test_dir, "MultiStructure.cif")], trans)
        self.assertEqual(len(tsc), 2)
        expected_ans = set(["Mn", "O", "Li", "P"])
        for s in tsc:
            els = set([el.symbol for el in s.final_structure.composition.elements])
            self.assertEqual(expected_ans, els)

    def test_poscar_transmute(self):
        trans = []
        trans.append(SubstitutionTransformation({"Fe":"Mn"}))
        tsc = TransformedStructureTransmuter.from_poscars([os.path.join(test_dir, "POSCAR"), os.path.join(test_dir, "POSCAR")], trans)
        self.assertEqual(len(tsc), 2)
        expected_ans = set(["Mn", "O", "P"])
        for s in tsc:
            els = set([el.symbol for el in s.final_structure.composition.elements])
            self.assertEqual(expected_ans, els)

    def test_transmuter(self):
        tsc = TransformedStructureTransmuter.from_poscars([os.path.join(test_dir, "POSCAR")])
        tsc.append_transformation(RemoveSpeciesTransformation('O'))
        self.assertEqual(len(tsc[0].final_structure), 8)

        tsc.append_transformation(SubstitutionTransformation({"Fe":{"Fe2+":.25, "Mn3+":.75}, "P":"P5+"}))
        tsc.append_transformation(OrderDisorderedStructureTransformation(), extend_collection=50)
        self.assertEqual(len(tsc), 4)

        t = SuperTransformation([SubstitutionTransformation({"Fe2+":"Mg2+"}), SubstitutionTransformation({"Fe2+":"Zn2+"})])
        tsc.append_transformation(t, extend_collection=True)
        self.assertEqual(len(tsc), 8)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()