#########################################################################################################
# Analysis published in:
# Simone Spolaor, Martijn Scheve, Murat Firat, Paolo Cazzaniga, Daniela Besozzi, Marco S. Nobile.
# Screening for combination cancer therapieswith dynamic fuzzy modeling andmulti-objective optimization.
# Frontiers in Genetics, 2021, 12: 449.
#########################################################################################################

from simpful import *
from copy import deepcopy
from collections import defaultdict
from platypus import NSGAII, Problem, Integer, ProcessPoolEvaluator
from numpy import savetxt, array, linspace
import matplotlib.pyplot as plt

class Model_Simulator(object):

	def __init__(self, steps=100):
		# Set simulation steps
		self._max_steps = steps

		self.FS = FuzzySystem()

		# Define linguistic variables
		GLUCOSE_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		GLUCOSE_2 = FuzzySet(points=[[0, 0], [0.3, 1], [0.7, 1], [1, 0]], term="Medium")
		GLUCOSE_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Glucose", LinguisticVariable([GLUCOSE_1, GLUCOSE_2, GLUCOSE_3], concept="Glucose"))

		GLYCOLYSIS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		GLYCOLYSIS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		GLYCOLYSIS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Glycolysis", LinguisticVariable([GLYCOLYSIS_1, GLYCOLYSIS_2, GLYCOLYSIS_3], concept="Glycolysis"))

		C1_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="LessFunctional")
		C1_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="MediumFunctional")
		C1_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="MoreFunctional")
		self.FS.add_linguistic_variable("C1", LinguisticVariable([C1_1, C1_2, C1_3], concept="C1"))

		DELTAPSI_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		DELTAPSI_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("DeltaPsi", LinguisticVariable([DELTAPSI_1, DELTAPSI_2], concept="DeltaPsi"))

		ROS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		ROS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		ROS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("ROS", LinguisticVariable([ROS_1, ROS_2, ROS_3], concept="ROS"))

		PKA_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		PKA_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("PKA", LinguisticVariable([PKA_1, PKA_2], concept="PKA"))

		ATP_1 = FuzzySet(points=[[0, 1], [0.1, 0]], term="VeryLow")
		ATP_2 = FuzzySet(points=[[0, 0], [0.1, 1], [0.4, 1], [0.5, 0]], term="Low")
		ATP_3 = FuzzySet(points=[[0.4, 0], [0.5, 1], [1, 0]], term="Medium")
		ATP_4 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("ATP", LinguisticVariable([ATP_1, ATP_2, ATP_3, ATP_4], concept="ATP"))

		CA2_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		CA2_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		CA2_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("CA2", LinguisticVariable([CA2_1, CA2_2, CA2_3], concept="CA2"))

		HBP_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		HBP_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		HBP_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("HBP", LinguisticVariable([HBP_1, HBP_2, HBP_3], concept="HBP"))

		NGLYCOS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		NGLYCOS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		NGLYCOS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("NGlycos", LinguisticVariable([NGLYCOS_1, NGLYCOS_2, NGLYCOS_3], concept="NGlycos"))

		UPR_1 = FuzzySet(points=[[0, 1], [0.2, 1], [0.6, 0]], term="Low")
		UPR_2 = FuzzySet(points=[[0.2, 0], [0.6, 1], [1, 0]], term="Medium")
		UPR_3 = FuzzySet(points=[[0.6, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("UPR", LinguisticVariable([UPR_1, UPR_2, UPR_3], concept="UPR"))

		CHOP_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		CHOP_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		CHOP_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("CHOP", LinguisticVariable([CHOP_1, CHOP_2, CHOP_3], concept="CHOP"))

		BCL2_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		BCL2_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		BCL2_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Bcl2", LinguisticVariable([BCL2_1, BCL2_2, BCL2_3], concept="Bcl2"))

		JNK_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		JNK_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("JNK", LinguisticVariable([JNK_1, JNK_2], concept="JNK"))

		AUTOPHAGY_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		AUTOPHAGY_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		AUTOPHAGY_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Autophagy", LinguisticVariable([AUTOPHAGY_1, AUTOPHAGY_2, AUTOPHAGY_3], concept="Autophagy"))

		DAPK_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		DAPK_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("DAPK", LinguisticVariable([DAPK_1, DAPK_2], concept="DAPK"))

		BCN1_1 = FuzzySet(points=[[0, 1], [0.33, 0]], term="Low")
		BCN1_2 = FuzzySet(points=[[0, 0], [0.33, 1], [0.66, 0]], term="Medium")
		BCN1_3 = FuzzySet(points=[[0.33, 0], [0.66, 1], [1.0, 0]], term="High")
		BCN1_4 = FuzzySet(points=[[0.66, 0], [1, 1]], term="VeryHigh")
		self.FS.add_linguistic_variable("BCN1", LinguisticVariable([BCN1_1, BCN1_2, BCN1_3, BCN1_4], concept="BCN1"))

		CASPASE3_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		CASPASE3_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		CASPASE3_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Caspase3", LinguisticVariable([CASPASE3_1, CASPASE3_2, CASPASE3_3], concept="Caspase3"))

		ATTACH_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		ATTACH_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Attach", LinguisticVariable([DAPK_1, DAPK_2], concept="Attach"))

		SRC_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		SRC_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Src", LinguisticVariable([SRC_1, SRC_2], concept="Src"))

		RASGTP_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Off")
		RASGTP_2 = FuzzySet(points=[[0, 0], [1, 1]], term="On")
		self.FS.add_linguistic_variable("RasGTP", LinguisticVariable([RASGTP_1, RASGTP_2], concept="RasGTP"))

		NECROSIS_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		NECROSIS_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Necrosis", LinguisticVariable([NECROSIS_1, NECROSIS_2], concept="Necrosis"))

		APOPTOSIS_1 = FuzzySet(points=[[0, 1], [0.5, 0]], term="Low")
		APOPTOSIS_2 = FuzzySet(points=[[0, 0], [0.5, 1], [1, 0]], term="Medium")
		APOPTOSIS_3 = FuzzySet(points=[[0.5, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Apoptosis", LinguisticVariable([APOPTOSIS_1, APOPTOSIS_2, APOPTOSIS_3], concept="Apoptosis"))

		SURVIVAL_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		SURVIVAL_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("Survival", LinguisticVariable([SURVIVAL_1, SURVIVAL_2], concept="Survival"))

		ERK_1 = FuzzySet(points=[[0, 1], [1, 0]], term="Low")
		ERK_2 = FuzzySet(points=[[0, 0], [1, 1]], term="High")
		self.FS.add_linguistic_variable("ERK", LinguisticVariable([ERK_1, ERK_2], concept="ERK"))


		# Define output crisp values
		self.FS.set_crisp_output_value("Low", 0)
		self.FS.set_crisp_output_value("Medium", 0.5)
		self.FS.set_crisp_output_value("High", 1)

		self.FS.set_crisp_output_value("VeryLow_ATP", 0)
		self.FS.set_crisp_output_value("Low_ATP", 0.1)

		self.FS.set_crisp_output_value("Medium_BCN1", 0.33)
		self.FS.set_crisp_output_value("High_BCN1", 0.66)
		self.FS.set_crisp_output_value("VeryHigh_BCN1", 1)

		self.FS.set_crisp_output_value("LessFunctional", 0)
		self.FS.set_crisp_output_value("MediumFunctional", 0.5)
		self.FS.set_crisp_output_value("MoreFunctional", 1)


		# Define fuzzy rules
		RULES=[]

		#Glycolysis_rules
		RULES.append("IF (Glucose IS Low) THEN (Glycolysis IS Low)")
		RULES.append("IF (Glucose IS Medium) THEN (Glycolysis IS Medium)")
		RULES.append("IF (Glucose IS High) THEN (Glycolysis IS High)")

		#C1_rules
		RULES.append("IF (PKA IS Low) THEN (C1 IS MediumFunctional)") 
		RULES.append("IF (Glycolysis IS High) AND (PKA IS High) THEN (C1 IS MoreFunctional)")
		RULES.append("IF (Glycolysis IS Medium) AND (PKA IS High) THEN (C1 IS MoreFunctional)")
		RULES.append("IF (Glycolysis IS High) AND (PKA IS Low) THEN (C1 IS MediumFunctional)")       
		RULES.append("IF (Glycolysis IS Medium) AND (PKA IS Low) THEN (C1 IS MediumFunctional)")  
		RULES.append("IF (Glycolysis IS Low) AND (PKA IS High) THEN (C1 IS MoreFunctional)")
		RULES.append("IF (Glycolysis IS Low) AND (PKA IS Low) THEN (C1 IS LessFunctional)")

		#DeltaPsi_rules
		RULES.append("IF (Glycolysis IS Medium) THEN (DeltaPsi IS High)")
		RULES.append("IF (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
		RULES.append("IF (C1 IS MoreFunctional) THEN (DeltaPsi IS High)")
		RULES.append("IF (C1 IS MediumFunctional) THEN (DeltaPsi IS High)")
		RULES.append("IF (CA2 IS Low) THEN (DeltaPsi IS High)")
		RULES.append("IF (Bcl2 IS High) THEN (DeltaPsi IS High)")
		RULES.append("IF (CA2 IS High) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
		RULES.append("IF (C1 IS LessFunctional) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
		RULES.append("IF (Bcl2 IS Low) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
		RULES.append("IF (CA2 IS Medium) AND (Glycolysis IS High) THEN (DeltaPsi IS High)")
		RULES.append("IF (CA2 IS High) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
		RULES.append("IF (CA2 IS High) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
		RULES.append("IF (C1 IS LessFunctional) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
		RULES.append("IF (C1 IS LessFunctional) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
		RULES.append("IF (Bcl2 IS Low) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
		RULES.append("IF (Bcl2 IS Low) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
		RULES.append("IF (CA2 IS Medium) AND (Glycolysis IS Medium) THEN (DeltaPsi IS Low)")
		RULES.append("IF (CA2 IS Medium) AND (Glycolysis IS Low) THEN (DeltaPsi IS Low)")
		RULES.append("IF (Bcl2 IS Medium) THEN (DeltaPsi IS Low)")
		RULES.append("IF (Glycolysis IS High) THEN (DeltaPsi IS High)")

		#ROS_rules
		RULES.append("IF (C1 IS LessFunctional) THEN (ROS IS High)")
		RULES.append("IF (C1 IS MediumFunctional) THEN (ROS IS Medium)")
		RULES.append("IF (C1 IS MoreFunctional) THEN (ROS IS Low)")
		RULES.append("IF (DeltaPsi IS Low) THEN (ROS IS High)")
		RULES.append("IF (DeltaPsi IS High) THEN (ROS IS Medium)")

		#ATP_rules
		RULES.append("IF (DeltaPsi IS High) THEN (ATP IS High)")
		RULES.append("IF (Glycolysis IS High) THEN (ATP IS High)")
		RULES.append("IF (Glycolysis IS Medium) THEN (ATP IS Medium)")
		RULES.append("IF (Glycolysis IS Low) THEN (ATP IS Low_ATP)")
		RULES.append("IF (C1 IS MoreFunctional) THEN (ATP IS High)")
		RULES.append("IF (Glycolysis IS High) AND (DeltaPsi IS Low) THEN (ATP IS High)") 
		RULES.append("IF (Glycolysis IS High) AND (C1 IS LessFunctional) THEN (ATP IS High)") 
		RULES.append("IF (Glycolysis IS High) AND (C1 IS MediumFunctional) THEN (ATP IS High)") 
		RULES.append("IF (Glycolysis IS Medium) AND (DeltaPsi IS Low) THEN (ATP IS Medium)") 
		RULES.append("IF (Glycolysis IS Medium) AND (C1 IS LessFunctional) THEN (ATP IS Medium)") 
		RULES.append("IF (Glycolysis IS Low) AND (DeltaPsi IS Low) THEN (ATP IS Medium)") 
		RULES.append("IF (Glycolysis IS Low) AND (C1 IS LessFunctional) THEN (ATP IS VeryLow_ATP)") 
		RULES.append("IF (Glycolysis IS Low) AND (C1 IS MediumFunctional) THEN (ATP IS Medium)") 
		RULES.append("IF (Glycolysis IS Medium) AND (C1 IS MediumFunctional) THEN (ATP IS Medium)") 

		#CA2_rules
		RULES.append("IF (UPR IS Low) THEN (CA2 IS Low)")
		RULES.append("IF (UPR IS Medium) THEN (CA2 IS Medium)")
		RULES.append("IF (UPR IS High) THEN (CA2 IS High)")

		#HBP_rules
		RULES.append("IF (Glucose IS High) AND (Autophagy IS Low) THEN (HBP IS High)")
		RULES.append("IF (Glucose IS High) AND (Autophagy IS Medium) THEN (HBP IS High)")
		RULES.append("IF (Glucose IS High) AND (Autophagy IS High) THEN (HBP IS High)")
		RULES.append("IF (Glucose IS Medium) AND (Autophagy IS Low) THEN (HBP IS Medium)")
		RULES.append("IF (Glucose IS Medium) AND (Autophagy IS Medium) THEN (HBP IS Medium)")
		RULES.append("IF (Glucose IS Medium) AND (Autophagy IS High) THEN (HBP IS High)")        
		RULES.append("IF (Glucose IS Low) AND (Autophagy IS Low) THEN (HBP IS Low)")
		RULES.append("IF (Glucose IS Low) AND (Autophagy IS Medium) THEN (HBP IS Medium)")
		RULES.append("IF (Glucose IS Low) AND (Autophagy IS High) THEN (HBP IS High)")

		#NGlycos_rules
		RULES.append("IF (HBP IS High) THEN (NGlycos IS High)")
		RULES.append("IF (HBP IS Medium) THEN (NGlycos IS Medium)")
		RULES.append("IF (HBP IS Low) THEN (NGlycos IS Low)")     
		RULES.append("IF (ATP IS High) THEN (NGlycos IS High)")
		RULES.append("IF (ATP IS Medium) THEN (NGlycos IS High)")
		RULES.append("IF (ATP IS Low) THEN (NGlycos IS Medium)")
		RULES.append("IF (ATP IS VeryLow) THEN (NGlycos IS Low)")

		#UPR_rules
		RULES.append("IF (NGlycos IS High) THEN (UPR IS Low)")
		RULES.append("IF (NGlycos IS Medium) THEN (UPR IS Medium)")
		RULES.append("IF (NGlycos IS Low) THEN (UPR IS High)")
		RULES.append("IF (ATP IS High) THEN (UPR IS Low)")
		RULES.append("IF (ATP IS Medium) THEN (UPR IS Low)") 
		RULES.append("IF (ATP IS Low) THEN (UPR IS Medium)")
		RULES.append("IF (ATP IS VeryLow) THEN (UPR IS High)")    

		#CHOP_rules
		RULES.append("IF (UPR IS Low) THEN (CHOP IS Low)")
		RULES.append("IF (UPR IS Medium) THEN (CHOP IS Medium)")
		RULES.append("IF (UPR IS High) THEN (CHOP IS High)")    

		#JNK_rules
		RULES.append("IF (UPR IS High) THEN (JNK IS High)")
		RULES.append("IF (UPR IS Medium) THEN (JNK IS High)")
		RULES.append("IF (UPR IS Low) THEN (JNK IS Low)")       

		#Autophagy_rules               
		RULES.append("IF (CA2 IS High) THEN (Autophagy IS High)")    
		RULES.append("IF (BCN1 IS High) THEN (Autophagy IS High)")
		RULES.append("IF (BCN1 IS VeryHigh) THEN (Autophagy IS High)")
		RULES.append("IF (ROS IS High) THEN (Autophagy IS High)")
		RULES.append("IF (Glycolysis IS Low) THEN (Autophagy IS High)")
		RULES.append("IF (CA2 IS Low) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (BCN1 IS Low) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (ATP IS VeryLow) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (ATP IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (ATP IS High) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (Glycolysis IS High) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (CA2 IS Low) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (BCN1 IS Low) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (ATP IS VeryLow) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (ATP IS Medium) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (ATP IS High) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (Glycolysis IS High) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (CA2 IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (BCN1 IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (Glycolysis IS Medium) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (CA2 IS Medium) AND (PKA IS Low) THEN (Autophagy IS Medium)")
		RULES.append("IF (BCN1 IS Medium) AND (PKA IS Low) THEN (Autophagy IS Medium)")
		RULES.append("IF (Glycolysis IS Medium) AND (PKA IS Low) THEN (Autophagy IS Medium)")
		RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
		RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
		RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
		RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Low)")
		RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Low)")
		RULES.append("IF (CA2 IS Low) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Low)")
		RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Low)")
		RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Low)")
		RULES.append("IF (BCN1 IS Low) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Low)")
		RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Low)")
		RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Low)")
		RULES.append("IF (Glycolysis IS High) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Low)")
		RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)")
		RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)") 
		RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS Low)) THEN (Autophagy IS High)") 
		RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Medium)")
		RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Medium)")
		RULES.append("IF (CA2 IS Medium) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Medium)")
		RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Medium)")
		RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Medium)")
		RULES.append("IF (BCN1 IS Medium) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Medium)")
		RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS VeryLow)) THEN (Autophagy IS Medium)")
		RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS Medium)) THEN (Autophagy IS Medium)")
		RULES.append("IF (Glycolysis IS Medium) AND ((PKA IS Low) AND (ATP IS High)) THEN (Autophagy IS Medium)")
		RULES.append("IF (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (ROS IS Medium) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (ROS IS Medium) AND (PKA IS High) THEN (Autophagy IS Medium)")
		RULES.append("IF (ROS IS Low) AND (PKA IS Low) THEN (Autophagy IS Low)")
		RULES.append("IF (ROS IS Low) AND (PKA IS High) THEN (Autophagy IS High)")
		RULES.append("IF (PKA IS High) AND (ATP IS Low) THEN (Autophagy IS High)")
		RULES.append("IF (PKA IS Low) AND (ATP IS Low) THEN (Autophagy IS Medium)")
		RULES.append("IF (PKA IS Low) AND (CA2 IS High) THEN (Autophagy IS High)")
		RULES.append("IF (PKA IS Low) AND (BCN1 IS High) THEN (Autophagy IS High)")
		RULES.append("IF (PKA IS Low) AND (Glycolysis IS Low) THEN (Autophagy IS High)")    

		#DAPK_rules
		RULES.append("IF (Src IS High) THEN (DAPK IS Low)")
		RULES.append("IF (CA2 IS Low) THEN (DAPK IS Low)")
		RULES.append("IF (ERK IS High) THEN (DAPK IS Low)")
		RULES.append("IF (Src IS Low) THEN (DAPK IS High)")
		RULES.append("IF (CA2 IS High) THEN (DAPK IS High)")
		RULES.append("IF (CA2 IS Medium) THEN (DAPK IS High)")
		RULES.append("IF (ERK IS Low) THEN (DAPK IS High)")

		#Bcl2_rules
		RULES.append("IF (CHOP IS High) THEN (Bcl2 IS Low)")
		RULES.append("IF (CHOP IS Medium) THEN (Bcl2 IS Medium)")
		RULES.append("IF (CHOP IS Low) THEN (Bcl2 IS High)")
		RULES.append("IF (CHOP IS Medium) AND (JNK IS High) THEN (Bcl2 IS Low)")
		RULES.append("IF (CHOP IS Low) AND (JNK IS High) THEN (Bcl2 IS Low)")
		RULES.append("IF (CHOP IS Medium) AND (JNK IS Low) THEN (Bcl2 IS Medium)")
		RULES.append("IF (BCN1 IS High) THEN (Bcl2 IS Low)")
		RULES.append("IF (ATP IS High) THEN (Bcl2 IS High)")
		RULES.append("IF (ATP IS Medium) THEN (Bcl2 IS High)")
		RULES.append("IF (ATP IS Low) THEN (Bcl2 IS Low)")
		RULES.append("IF (ATP IS VeryLow) THEN (Bcl2 IS Low)")
		RULES.append("IF (JNK IS Low) THEN (Bcl2 IS High)")
		RULES.append("IF (BCN1 IS Low) THEN (Bcl2 IS High)")
		RULES.append("IF (CHOP IS High) AND (JNK IS High) THEN (Bcl2 IS Low)") 
		RULES.append("IF (BCN1 IS Medium) THEN (Bcl2 IS Medium)")
		RULES.append("IF (BCN1 IS VeryHigh) THEN (Bcl2 IS Low)")
		RULES.append("IF (PKA IS High) THEN (Bcl2 IS Medium)")
		RULES.append("IF (PKA IS Low) THEN (Bcl2 IS High)")            

		#BCN1_rules
		RULES.append("IF (Bcl2 IS Low) THEN (BCN1 IS High_BCN1)")
		RULES.append("IF (Bcl2 IS Medium) THEN (BCN1 IS Medium_BCN1)")
		RULES.append("IF (Bcl2 IS High) THEN (BCN1 IS Low)")        
		RULES.append("IF (DAPK IS High) THEN (BCN1 IS VeryHigh_BCN1)")
		RULES.append("IF (DAPK IS Low) THEN (BCN1 IS Low)")
		    
		#Caspase3_rules
		RULES.append("IF (ATP IS Low) THEN (Caspase3 IS High)")
		RULES.append("IF (ATP IS High) THEN (Caspase3 IS Low)")
		RULES.append("IF (ATP IS Medium) THEN (Caspase3 IS Low)")
		RULES.append("IF (ROS IS Low) THEN (Caspase3 IS Low)")
		RULES.append("IF (ROS IS Medium) THEN (Caspase3 IS Low)")
		RULES.append("IF (CA2 IS Medium) THEN (Caspase3 IS Low)")
		RULES.append("IF (CA2 IS Low) THEN (Caspase3 IS Low)")
		RULES.append("IF (ERK IS High) THEN (Caspase3 IS Low)")
		RULES.append("IF (Bcl2 IS High) THEN (Caspase3 IS Low)")
		RULES.append("IF (DeltaPsi IS High) THEN (Caspase3 IS Low)")
		RULES.append("IF (ROS IS High) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
		RULES.append("IF (CA2 IS High) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
		RULES.append("IF (DeltaPsi IS Low) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)") 
		RULES.append("IF (ROS IS High) AND (ATP IS High) THEN (Caspase3 IS High)")
		RULES.append("IF (ROS IS High) AND (ATP IS Medium) THEN (Caspase3 IS High)")
		RULES.append("IF (ROS IS High) AND (ATP IS Low) THEN (Caspase3 IS High)")
		RULES.append("IF (CA2 IS High) AND (ATP IS High) THEN (Caspase3 IS High)")
		RULES.append("IF (CA2 IS High) AND (ATP IS Medium) THEN (Caspase3 IS High)")
		RULES.append("IF (CA2 IS High) AND (ATP IS Low) THEN (Caspase3 IS High)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS High) THEN (Caspase3 IS High)")        
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS Medium) THEN (Caspase3 IS High)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS Low) THEN (Caspase3 IS High)")
		RULES.append("IF (DeltaPsi IS Low) AND (ATP IS High) THEN (Caspase3 IS High)")
		RULES.append("IF (DeltaPsi IS Low) AND (ATP IS Medium) THEN (Caspase3 IS High)")
		RULES.append("IF (DeltaPsi IS Low) AND (ATP IS Low) THEN (Caspase3 IS High)")
		RULES.append("IF (Bcl2 IS Medium) AND (ATP IS VeryLow) THEN (Caspase3 IS Low)")
		RULES.append("IF (Bcl2 IS Medium) AND (ATP IS High) THEN (Caspase3 IS Medium)")
		RULES.append("IF (Bcl2 IS Medium) AND (ATP IS Medium) THEN (Caspase3 IS Medium)")
		RULES.append("IF (Bcl2 IS Medium) AND (ATP IS Low) THEN (Caspase3 IS Medium)")
		RULES.append("IF (ATP IS VeryLow) THEN (Caspase3 IS Low)")
		RULES.append("IF (ERK IS Low) THEN (Caspase3 IS Medium)")
		    
		#Attach_rules
		RULES.append("IF (NGlycos IS High) THEN (Attach IS High)")
		RULES.append("IF (NGlycos IS Low) THEN (Attach IS Low)")
		RULES.append("IF (NGlycos IS Medium) THEN (Attach IS Low)")

		#Src_rules
		RULES.append("IF (Attach IS Low) THEN (Src IS Low)")
		RULES.append("IF (Attach IS High) THEN (Src IS High)")
		RULES.append("IF (PKA IS High) THEN (Src IS High)")
		RULES.append("IF (PKA IS Low) THEN (Src IS Low)")
		RULES.append("IF (PKA IS High) AND (Attach IS Low) THEN (Src IS High)")    

		#ERK_rules
		RULES.append("IF (Src IS High) THEN (ERK IS High)")
		RULES.append("IF (Src IS Low) AND (RasGTP IS On) THEN (ERK IS High)")
		RULES.append("IF (DAPK IS High) AND (RasGTP IS On) THEN (ERK IS High)")
		RULES.append("IF (Src IS Low) AND (RasGTP IS Off) THEN (ERK IS Low)")
		RULES.append("IF (DAPK IS High) AND (RasGTP IS Off) THEN (ERK IS Low)")
		RULES.append("IF (RasGTP IS On) THEN (ERK IS High)")
		RULES.append("IF (DAPK IS Low) THEN (ERK IS High)")
		RULES.append("IF (RasGTP IS Off) AND (Src IS High) THEN (ERK IS High)")
		RULES.append("IF (RasGTP IS Off) AND (DAPK IS Low) THEN (ERK IS High)") 

		#Necrosis_rules
		RULES.append("IF (Bcl2 IS High) THEN (Necrosis IS Low)")
		RULES.append("IF (Bcl2 IS Medium) THEN (Necrosis IS Low)")
		RULES.append("IF (ATP IS VeryLow) THEN (Necrosis IS High)")
		RULES.append("IF (ATP IS Low) THEN (Necrosis IS Low)")
		RULES.append("IF (ATP IS Medium) THEN (Necrosis IS Low)")
		RULES.append("IF (ROS IS Medium) THEN (Necrosis IS Low)")
		RULES.append("IF (ROS IS Low) THEN (Necrosis IS Low)")
		RULES.append("IF (ROS IS High) AND (ATP IS High) THEN (Necrosis IS Low)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS High) THEN (Necrosis IS Low)")
		RULES.append("IF (ROS IS High) AND (ATP IS VeryLow) THEN (Necrosis IS High)")
		RULES.append("IF (ROS IS High) AND (ATP IS Low) THEN (Necrosis IS High)")
		RULES.append("IF (ROS IS High) AND (ATP IS Medium) THEN (Necrosis IS High)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS VeryLow) THEN (Necrosis IS High)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS Low) THEN (Necrosis IS High)")
		RULES.append("IF (Bcl2 IS Low) AND (ATP IS Medium) THEN (Necrosis IS High)")
		RULES.append("IF (ATP IS High) THEN (Necrosis IS Low)")       

		#Apoptosis_rules
		RULES.append("IF (Caspase3 IS Low) THEN (Apoptosis IS Low)")
		RULES.append("IF (Caspase3 IS Medium) THEN (Apoptosis IS Medium)")
		RULES.append("IF (Caspase3 IS High) THEN (Apoptosis IS High)")       

		#Survival_rules
		RULES.append("IF (DeltaPsi IS Low) THEN (Survival IS Low)")
		RULES.append("IF (DeltaPsi IS High) THEN (Survival IS High)")
		RULES.append("IF (Caspase3 IS High) THEN (Survival IS Low)")
		RULES.append("IF (Caspase3 IS Low) THEN (Survival IS High)")
		RULES.append("IF (Caspase3 IS Medium) THEN (Survival IS Low)")
		RULES.append("IF (ATP IS VeryLow) THEN (Survival IS Low)")
		RULES.append("IF (ATP IS Low) THEN (Survival IS Low)")
		RULES.append("IF (ATP IS Medium) THEN (Survival IS High)")
		RULES.append("IF (ATP IS High) THEN (Survival IS High)")
		RULES.append("IF (Autophagy IS Medium) THEN (Survival IS High)")
		RULES.append("IF (Autophagy IS High) THEN (Survival IS High)")
		RULES.append("IF (Autophagy IS Low) THEN (Survival IS Low)") 

		self.FS.add_rules(RULES)

		# Set initial state of the model
		self._reset_variables()    

		# Set variables to be perturbed
		self._sorted_names = sorted(list(self.FS._variables))
		self._sorted_names.remove("Apoptosis")
		self._sorted_names.remove("Caspase3")
		self._sorted_names.remove("Survival")
		self._sorted_names.remove("Necrosis")
		self._sorted_names.remove("Glucose")
		self._sorted_names.remove("PKA")
		self._sorted_names.remove("RasGTP")
		self._sorted_names.remove("ATP")
		self._sorted_names.remove("Glycolysis")
		print(" * Variables being perturbed:", self._sorted_names)
        
	def _reset_variables(self):
		self.FS.set_variable("Glucose", 1.0)
		self.FS.set_variable("Glycolysis", 1.0)
		self.FS.set_variable("C1", 0.5)
		self.FS.set_variable("DeltaPsi", 1.0)
		self.FS.set_variable("ROS", 0.5)
		self.FS.set_variable("ATP", 1.0)
		self.FS.set_variable("CA2", 0.0)
		self.FS.set_variable("HBP", 1.0)
		self.FS.set_variable("NGlycos", 1.0)
		self.FS.set_variable("UPR", 0.0)
		self.FS.set_variable("CHOP", 0.0)
		self.FS.set_variable("Bcl2", 1.0)
		self.FS.set_variable("JNK", 0.0)
		self.FS.set_variable("Autophagy", 0.0)
		self.FS.set_variable("DAPK", 0.0)
		self.FS.set_variable("BCN1", 0.0)
		self.FS.set_variable("Caspase3", 0.0)
		self.FS.set_variable("Attach", 1.0)
		self.FS.set_variable("Src", 1.0)
		self.FS.set_variable("Necrosis", 0.0)
		self.FS.set_variable("Apoptosis", 0.0)
		self.FS.set_variable("Survival", 1.0)
		self.FS.set_variable("ERK", 1.0)
		self.FS.set_variable("PKA", 1.0)
		self.FS.set_variable("RasGTP", 1.0)


	def simulate(self, perturbation=None):
		# Simulate the model with a perturbation
		dynamics = defaultdict(list)
		self._reset_variables()
		for T in linspace(0, 1, self._max_steps):
		    self.FS.set_variable("Glucose", time_function(T))
		    
		    for k,v in zip(self._sorted_names, perturbation):
		    	if v==1: #low
		    		self.FS.set_variable(k, 0.0)		    		
		    	elif v==2: #high
		    		self.FS.set_variable(k, 1.0)

		    new_values = self.FS.Sugeno_inference()

		    self.FS._variables.update(new_values)

		    for var in new_values.keys():          
		        dynamics[var].append(new_values[var])

		return dynamics

	def fitness(self, x, SIM=None):
		# Calculate fitness of the perturbation
		result = self.simulate(perturbation=x)
		begin_apo = result['Apoptosis'][0]
		end_apo = result['Apoptosis'][14] # t>0.13
		complexity = len(list(filter(lambda y: y>0, x)))
		print(x, "%.3f, %d" % (end_apo-begin_apo, complexity))
		return end_apo-begin_apo, complexity

def time_function(curtime):
	if curtime<0.075: 
		return 1
	elif curtime>0.7: 
		return 0
	else: 
		return 1./(7*curtime**0.75)-0.185

def is_pareto_efficient(costs, return_mask=True):
    # Find the pareto-efficient points
    import numpy as np
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


if __name__ == '__main__':
	
	SIM = Model_Simulator()

	D = len(SIM._sorted_names)
	OBJS = 2
	FEs = 10000
	POPSIZE = 100
	DIRECTIONS = [Problem.MAXIMIZE, Problem.MINIMIZE]

	print(" * %d variables, %d objectives" % (D,OBJS))
	print(" * %d individuals, MAX_FEs %d, %d iterations" % (POPSIZE, FEs, FEs//POPSIZE)) 
	problem = Problem(D, OBJS)
	all_types = [Integer(0,2) for _ in range(D)]
	problem.types[:] = all_types
	problem.function = SIM.fitness
	problem.directions[:] = DIRECTIONS
	
	# Multi-processing (4 parallel processes)
	with ProcessPoolEvaluator(4) as evaluator:
		algorithm = NSGAII(problem, population_size = POPSIZE, evaluator=evaluator)
		res = algorithm.run(FEs)

	final_results = [(s.objectives[0], s.objectives[1]) for s in algorithm.result]
	final_solutions = []
	for solution in algorithm.result:
		temp = []
		for n in range(D):
			tpe = all_types[n]
			value = tpe.decode(solution.variables[n])
			name = SIM._sorted_names[n]
			if value==1:
				temp.append("%s IS low" % name)
			elif value==2:
				temp.append("%s IS high" % name)
		final_solutions.append(temp)
	print(final_results)
	print(final_solutions)
	savetxt("result-fitness.txt", final_results)
	with open("result-paretofront.txt", "w") as fo:
		for final_item in final_solutions:
			fo.write(", ".join(final_item)+"\n")

	to_be_checked = array([s.objectives for s in algorithm.result])
	print(to_be_checked)

	for n,opt_direction in enumerate(DIRECTIONS):
		if opt_direction==Problem.MAXIMIZE:
			to_be_checked.T[n]*=-1
	
	pareto_front = is_pareto_efficient(to_be_checked,  return_mask = True)
	print(pareto_front)

	plt.scatter([s.objectives[0] for s in array(algorithm.result)[pareto_front]],
	            [s.objectives[1] for s in array(algorithm.result)[pareto_front]],
	            label="Pareto front", color="red")

	plt.xlabel("Change in Apoptosis")
	plt.ylabel("Complexity of perturbation")
	plt.legend()
	plt.show()
