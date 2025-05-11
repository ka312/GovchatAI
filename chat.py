import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import json
import streamlit as st
import datetime
import re
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# ------------------- PostgreSQL Configuration -------------------
# Database connection parameters retrieved from environment variables
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# ------------------- Table Schema and Sample Data -------------------
# Target table name in the PostgreSQL database
TABLE_NAME = "tm_awards"

# Dictionary of column names and their descriptions, including data types
# Used to provide schema information to the AI model for query generation
COLUMN_DEFINITIONS = {
    "active_task_order": "If active_task_order value is 0 then task order is expired else active if not 0. NumberInt 32",
    "agency_entity_level": "Agency level is a term that describes the report system in corporate structures - the hierarchy of who reports to who. NumberInt 32",
    "award_id": "An award ID is a unique identification number representing each individual award. Some types of awards include grants, loans, direct payments, insurance, or contracts. String",
    "awarding_agency_name": "The name associated with a department or establishment of the Government as used in the Treasury Account Fund Symbol (TAFS). String",
    "awarding_agency_office_name": "Name of the level n organization that awarded, executed or is otherwise responsible for the transaction. String",
    "awarding_agency_subtier_agency_code": "Identifier of the level 2 organization that awarded, executed or is otherwise responsible for the transaction. String",
    "awarding_agency_toptier_agency_code": "A department or establishment of the Government as used in the Treasury Account Fund Symbol (TAFS). String",
    "awarding_office_code": "Identifier of the level n organization that awarded, executed or is otherwise responsible for the transaction. String",
    "awarding_sub_agency_name": "Name of the level 2 organization that awarded, executed or is otherwise responsible for the transaction. String",
    "base_exercised_options": "The contract value for the base contract and any options that have been exercised. Dollar",
    "base_and_all_options": "For the Award it is the mutually agreed upon total contract value including all options (if any). For IDVs the value is the mutually agreed upon total contract value including all options (if any) AND the estimated value of all potential orders. Dollar",
    "cage_code": "CAGE stands for Commercial and Government Entity. A CAGE code is a unique 5-character identifier that provides a standardized method for recognizing facilities and their locations. String",
    "commercial_item_acquisition_standards": "Standards for commercial item acquisitions. String",
    "congressional_code": "Congressional district code. String",
    "contracting_offivers_determination_of_business_size": "Determination of business size by the contracting officer. String",
    "cost_or_pricing_data_code": "A designator that indicates if cost or pricing was obtained, not obtained or waived. String",
    "cost_or_pricing_data": "Description tag (by way of the FPDS Atom Feed) that explains the meaning of the code provided in the Cost or Pricing Data Field. String",
    "country_code": "Code for the country in which the awardee or recipient is located, using the International Standard for country codes (ISO) 3166-1 Alpha-3. String",
    "country_name": "The name corresponding to the country code. String",
    "date_signed": "The date signed marks the date that the contract was officially signed. ISO Date",
    "description": "The award title is submitted by the contracting officer and describes the funding provided. String",
    "dod_acquisition_program_code": "Two codes that identify the DoD program and system purchased. String",
    "dod_acquisition_program_description": "Explains the meaning of the code in the DOD Acquisition Program field. String",
    "end_date": "The end date marks the end of the contract's period of performance. ISO Date",
    "extent_compete_description": "Explains the meaning of the code provided in the Extent Competed Field. String",
    "extent_competed": "A code that represents the competitive nature of the contract. String",
    "fair_opportunity_limited_sources": "Explains the code in the Fair Opportunity Limited Sources Field. String",
    "funding_agency_office_name": "Name of the office that provided the funds. String",
    "funding_agency_subtier_agency_code": "Identifier of the funding agency's level 2 organization. String",
    "funding_agency_technomile_id": "TechnoMile GovSearch ID for the funding agency. NumberInt 64",
    "funding_agency_technomile_name": "TechnoMile GovSearch name for the funding agency. String",
    "funding_agency_toptier_agency_code": "Department code used in the Treasury Account Fund Symbol (TAFS). String",
    "funding_office_code": "Identifier of the funding office. String",
    "funding_sub_agency_name": "Name of the funding sub-agency. String",
    "generated_unique_award_id": "Derived unique key for the award (concatenation of various identifiers). String",
    "information_technology_commercial_item_category_code": "Designates the commercial availability of an IT product or service. String",
    "labor_standards": "Indicates whether the transaction is subject to labor standards. String",
    "last_modified_date": "The last modified date of the record. ISO Date",
    "location_country_code": "Country code for the awardee's location. String",
    "multi_year_contract": "Indicator and description for a multi-year contract. String",
    "multi_year_contract_code": "Code representing multi-year contract details. String",
    "naics": "6-digit NAICS code representing the industry. NumberInt 32",
    "naics_description": "The title describing the NAICS code. String",
    "national_interest_action": "Explains the meaning of the code in the National Interest Action Field. String",
    "national_interest_action_code": "Code representing the national interest for the contract. String",
    "number_of_actions": "Number of actions reported in one modification. NumberInt 32",
    "number_of_offers_received": "The number of offers received. String",
    "other_than_full_and_open_competition_code": "Designator for non-full and open competition procedures. String",
    "parent_award_piid": "Unique contract number for the parent award. String",
    "parent_award_single_or_multiple": "Indicates if the parent award is single or multiple. String",
    "parent_award_type": "Description of the parent award type. String",
    "parent_award_type_code": "Type code for the parent award. String",
    "parent_recipient_name": "Name of the parent recipient or vendor. String",
    "performance_based_service_acquisition": "Details of performance-based service acquisition. String",
    "piid": "Unique task order ID for the individual task order. String",
    "primary_place_of_performance_city_name": "City name where performance takes place. String",
    "primary_place_of_performance_zip_4": "ZIP code (with +4) for the performance location. String",
    "product_or_service_code": "4-digit code identifying the product or service (PSC). String",
    "product_or_service_code_description": "Description of the product or service code. String",
    "program_acronym": "Short name for a contracting program (e.g., COMMITS, ITOPS). String",
    "recipient_name": "Name of the awardee or recipient. String",
    "recipient_parent_uei": "Unique identifier for the parent vendor. String",
    "recipient_uei": "Unique identifier for the recipient. String",
    "recovered_materials_sustainability": "Indicates whether recovered materials clauses were included. String",
    "solicitation_identifier": "Identifier linking transactions to solicitation information. String",
    "solicitation_procedure_description": "Explains the meaning of the code in the Solicitation Procedures Field. String",
    "solicitation_procedures": "Designator for competitive solicitation procedures. String",
    "start_date": "The date when the contract goes into effect. ISO Date",
    "state_code": "USPS two-letter state code for the recipient's legal address. String",
    "state_name": "Name of the state where performance occurs. String",
    "subcontracting_plan": "Subcontracting plan requirement per FAR Part 19.702. String",
    "total_obligation": "Total amount of money obligated for the award. Dollar",
    "type_description": "Description of the contract action type. String",
    "type_of_contract_pricing": "Type of contract pricing per FAR Part 16. String",
    "type_of_contract_pricing_code": "Code representing the contract pricing type. String",
    "type_of_set_aside": "Indicates the type of set aside for the contract. String",
    "type_of_set_aside_code": "Code for the type of set aside determined for the contract action. String"
}

SAMPLE_DATA = {
    "active_task_order": ["0", "1"],
    "agency_entity_level": ["1", "2"],
    "award_id": ["ABCD1234", "XYZ5678", "TEST0001"],
    "awarding_agency_name": ["National Endowment for the Arts",
"Department of Justice",
"Department of Energy",
"HOMELAND SECURITY, DEPARTMENT OF",
"Department of the Treasury",
"Federal Election Commission",
"National Archives and Records Administration",
"Department of Defense",
"National Gallery of Art",
"Department of Housing and Urban Development",
"Department of Transportation",
"DEPT OF DEFENSE",
"Court Services and Offender Supervision Agency",
"Government Accountability Office",
"Commodity Futures Trading Commission",
"Department of the Interior",
"Department of Homeland Security",
"INTERIOR, DEPARTMENT OF THE",
"United States Chemical Safety Board",
"Environmental Protection Agency",
"GOVERNMENT ACCOUNTABILITY OFFICE (GAO)",
"Selective Service System",
"ENERGY, DEPARTMENT OF",
"Federal Communications Commission",
"Department of Agriculture",
"Consumer Financial Protection Bureau",
"Consumer Product Safety Commission",
"General Services Administration",
"Social Security Administration",
"U.S. Agency for Global Media",
"National Endowment for the Humanities"],
    "awarding_agency_office_name": ["TOOELE ARMY DEPOT CONTRACTING OF",
"DODEA EUROPE REGION OFFICE",
"DLA TROP SUPPORT C&E HARDWARE",
"PACIFIC MISSILE RANGE FACILITY",
"CONTRACTING&GENERAL SERVICES DIV.",
"U.S. ARMY CENTRAL COMMAND, QATAR",
"SOCOM/SOAL-KB",
"OFFICE OF CONTRACTS",
"SUPERFUND/RCRA REGIONAL PROCUREMENT OPERATIONS DIVISION (SRRPOD)",
"PEARL HARBOR NAVAL SHIPYARD IMF",
"TACOM - TEXARKANA",
"NAVAL SURFACE WARFARE CENTER",
"COMMODITY FUTURES TRADING COMMISSION",
"ACA, ITEC4",
"DITCO, SCOTT",
"DISA, NCR",
"ACA, 5TH SIGNAL COMMAND",
"SIERRA ARMY DEPOT",
"UNASSIGNED",
"CONSUMER PRODUCT SAFETY COMMISSION",
"NATIONAL GALLERY OF ARTS",
"DIVISION OF PROCUREMENT SERVICES",
"FEDERAL ELECTION COMMISSION",
"W071 ENDIST KANSAS CITY",
"DEPT OF INTER/BUREAU OF INDIAN AFFAIRS",
"ALBUQUERQUE ACQUISITION OFFICE",
"ACC-ABERDEEN PROVING GROUNDS CONTR",
"NATIONAL GEOSPATIAL TECHNICAL OPERATIONS CENTER III",
"LETTERKENNY ARMY DEPOT",
"REGION 4: EMERGENCY PREPAREDNESS AND RESPONSE",
"DITCO-EUROPE",
"W0L6 USA DEP LETTERKENY",
"EASTERN OKLAHOMA REGION",
"PORTSMOUTH NAVAL SHIPYARD",
"65 CONS/LGC",
"USA MATERIEL COMMAND ACQUISITION",
"OFFICE OF THE SPEC TRUSTEE",
"FA8819  HQ SMC SY PKS",
"W6QK ACC-APG",
"TELECOMMUNICATIONS DIVISION- HC1013",
"CENTER FOR COASTAL AND MARINE GEOLOGY",
"6913G6 VOLPE NATL. TRANS. SYS CNTR",
"OFFICE OF ACQUISITION AND GRANTS MANAGEMENT",
"COURT SERVICES & OFFENDER SUPERVISON AGENCY",
"OFFICE OF NAVAL RESEARCH, HEADQU",
"OFFICE OF ACQUISITION MANAGEMENT - WASHINGTON, DC OFFICE",
"SOCIAL SECURITY ADMINISTRATION",
"OFFICE OF ENVIRONMENTAL MANAGEMENT CONSOLIDATED BUSINESS CENTER",
"CFPB PROCUREMENT",
"CONSTRUCTION AND ACQUISITON DIVISION",
"W07V ENDIST ROCK ISLAND",
"DO NOT USE--REGION 05 - OFFICE OF THE REGIONAL COMMISSIONER",
"W6QK ACC-RSA",
"U.S. ARMY INDUSTRIAL OPERATIONS",
"DLA CONTRACTING SERVICES OFFICE",
"OAKLAND OPERATIONS OFFICE",
"OFFICE OF ACQUISITION AND GRANTS - RESTON",
"ACA, ABERDEEN PROVING GROUND",
"WR-ALC/PKO",
"W6QK PBA CONTR OFF",
"US GAO ACQUISITION MANAGEMENT",
"W6QK ACC-APG ADELPHI",
"NAVOPSPTCEN KNOXVILLE",
"US ARMY ROBERT MORRIS ACQUISITIO",
"W07V ENDIST N ORLEANS",
"FLAGSTAFF SCIENCE CENTER",
"DLA TROOP SUPPORT",
"CHUGACH NATIONAL FOREST",
"NAVAL SURFACE WARFARE CENTER CAR",
"SUPPLY & LOGISTICS OP,PURCHASE D",
"ACC-ABERDEEN PROVING GROUNDS CONT C",
"USPFO FOR TENNESSEE",
"SAN ANTONIO ALC/LDK",
"OFFICE OF ACQUISITION AND GRANTS - DENVER",
"USA AVIATION AND MISSILE COMMAND",
"W2SD ENDIST BALTIMORE",
"USA ENGINEER DISTRICT, BALTIMORE",
"READINESS&OPERATIONS BRANCH",
"FA7000  10 CONS LGC",
"DEPT HUD-CHIEF PROCUREMENT OFFICER",
"U S ARMY DEPOT RED RIVER",
"TACOM - PICATINNY",
"NAVAL REGIONAL CONTRACTING CENTE",
"DLA SUPPORT SERVICES - DSS",
"NTL INTERAGENCY FIRE CENTER",
"PSB 2",
"DEPT OF TRANS/COAST GUARD",
"DEPT OF TRANS/FEDERAL TRANSIT ADMINISTRATION",
"MILITARY SEALIFT COMMMAND",
"TACOM - ANNISTON",
"SAVANNAH RIVER OPERATIONS OFFICE",
"CEU JUNEAU",
"CHICAGO SERVICE CENTER (OFFICE OF SCIENCE)",
"VOA DIRECTORS OFFICE",
"OFFICE OF ACQUISITION AND GRANTS - SACRAMENTO",
"GSA/FAS ASSISTED AND EXPANDED ACQUISITION (R05)",
"W0LX ANNISTON DEPOT PROP DIV",
"IAS21WG",
"NAVAL COMMAND CONTROL & OCEAN SU",
"W6QK ADAP SPT OFF",
"FOREST SERVICE 120",
"NAVAL SEA SYSTEMS COMMAND",
"UNITED STATES MILITARY ACADEMY",
"WESTERN ADMIN. SERVICE CENTER  AVC",
"DC PRETRIAL SERVICES AGENCY",
"NAVOPSPTCEN SHREVEPORT",
"USA ENGINEER DISTRICT,",
"MID-WEST REGIONAL OFFICE",
"DODEA HQ",
"SPACE AND NAVAL WARFARE SYSTEMS",
"CENTRAL OFFICE",
"FEDERAL COMMUNICATIONS COMMISSION",
"DEPT OF TRANS/FEDERAL AVIATION ADMIN",
"DEPT OF TRANS/FEDERAL RAILROAD ADMIN",
"DLA TROOP SUPPORT C&E (MAT&ME)",
"ALBUQUERQUE SEISMOLOGICAL LAB",
"TOOELE ARMY DEPOT",
"AR STATE OFFICE (NRCS)",
"HQ USAINSCOM, DIR OF CONTRACTING",
"W40M EUR REGIONAL CONTRACT OFC",
"NMC CONUS WEST DIVISION",
"NAT'L ENDOWMENT FOR THE HUMANITIES",
"FAR EAST CONTRACTS OFFICE",
"NEVADA OPERATIONS OFFICE",
"MID-WEST REGION",
"WIESBADEN REGIONAL CONTRNG. CTR.",
"ROCKY MOUNTAIN REGION",
"DEPT OF TRANS/NAT HIGHWAY TRAFFIC SAFETY ADM",
"NAVAL FACILITIES ENGINEER COM",
"USPFO FOR IDAHO",
"COMMANDER",
"FISC NORFOLK NAVAL SHIPYARD ANNE",
"DIR OF SUB  DLA TROOP SUPPORT",
"CONTRACTING AND FACILITIES MGMT DIV",
"W4MM USA JOINT MUNITIONS CMD",
"CONSUMER FINANCE PROTECTION BUREAU",
"FA8620  AFLCMC WI",
"WATERVLIET ARSENAL",
"USA OSC MCALESTER ARMY AMMO PLNT",
"WSMR",
"TACOM - WARREN",
"HEADQUARTERS PROCUREMENT SERVICES",
"HQ DEF CONTRACT MANAGEMENT AGENCY",
"SELECTIVE SERVICE SYSTEM",
"WESTERN ADMINISTRATIVE SERVICE CENTER",
"NAVAL INVENTORY CONTROL POINT",
"CHEMICAL SAFETY AND HAZARD INVESTIGATION BOARD",
"U S ARMY DEPOT TOBYHANNA",
"INTERIOR FRANCHISE FUND",
"374 CONS/LGC MGMT ANAL & SPT",
"USSOCOM REGIONAL CONTRACTING OFFICE",
"45 CONS/LGC",
"ACA, FORT LEWIS",
"DEF REUTILIZATION & MARKETING SV",
"NAVSUP WEAPON SYSTEMS SUPPORT",
"NEW ORLEANS OFFICE",
"U.S. ARMY COMBINED ARMS CTR & FT",
"ACQUISITION SERVICES DIVISION",
"ACQUISITION SERVICES DIVISION",
"UNKNOWN OFFICE NAME",
"ALBUQUERQUE OPERATIONS OFFICE",
"DIRECTORATE OF CONTRACTING",
"NROTCU UNIVERSITY OF FLORIDA",
"DODEA PACIFIC",
"RICHLAND OPERATIONS OFFICE",
"ACQUISITION MANAGEMENT DIVISION",
"AGRIC STABILIZ AND CONS SVC",
"ROCK ISLAND ARSENAL",
"PROCUREMENT & SUPPORT SERVICES DIV./FEDSIM",
"DLA  ENERGY",
"W6QM MICC-WEST POINT",
"W00Y CONTR OFC DODAAC",
"ACA, U.S. MILITARY ACADEMY",
"DLA SUPPORT SERVICES",
"DEFENSE SUPPLY CENTER COLUMBUS",
"DEFENSE SUPPLY CENTER PHILADELPH",
"MDA",
"DODDS EUROPE  DIRECTOR'S OFFICE",
"OFFICE OF EXTERNAL AFFAIRS",
"OFFICE OF NAVAL RESEARCH",
"OFFICE OF ACQUISITION AND GRANTS - NATIONAL",
"DEFENSE SECURITY COOPERATION AGENCY",
"DEPT OF TREAS/U.S. MINT",
"NAVAJO REGION",
"NBC ACQUISITION SERVICES DIVISION",
"CORPUS CHRISTI ARMY DEPOT",
"NAVAL SURFACE WARFARE CENTER, PO",
"NAVAL AIR WARFARE CNETER TRAININ",
"NARA CONTRACTING OFFICE",
"DEFENSE INDUSTRIAL SUPPLY CENTER",
"NAVOPSPTCEN WHDBY ISLAND",
"ACC - ARSENALS, DEPOTS AND AMMO PLA",
"MARINE CORPS SYSTEMS COMMAND",
"W6QK ACC-APG CONT CT WASH OFC",
"PSB 3",
"DODDS EUROPEAN PROCUREMENT OFFICE",
"IBC ACQUISITION SERVICES DIVISION",
"NBC ACQUISITION SERVICES DIRECTORATE",
"NATIONAL ENDOWMENT FOR THE ARTS",
"USA COMMUNICATIONS-ELECTRONICS",
"NAVAL SURFACE WARFARE CENTER, IN",
"W3ZL OFC PM SANG MOD PROG",
"NATIONAL NUCLEAR SECURITY ADMN BUSINESS SVCS DIVISION",
"DLA STRATEGIC MATERIALS",
"W4GG HQ US ARMY TACOM",
"PSB 1",
"CONTRACTING AND GENERAL SERVICES DIV",
"ANNEX",
"GREAT PLAINS REGION",
"FA8818  HQ SMC SD PKT",
"W6QK ACC-APG NATICK",
"SOUTHERN PLAINS REGION",
"NAVFAC EFA SOUTHEAST ENGINEERING",
"DOT-OR&T 00057",
"BOARD OF GOVERNORS",
"W6QM MICC-DUGWAY PROV GRD",
"FLEET & INDUSTRIAL SUPPLY CENTER",
"NSWC CRANE",
"W7NY USPFO ACTIVITY RI ARNG",
"NAVAL FAC ENGINEEERING CMD EUR SWA",
"AVIATION APPLIED TECHNOLOGY",
"W6QK SIAD CONTR OFF",
"KY STATE OFFICE (NRCS)",
"MDW, FORT A. P. HILL",
"USA WATERVLIET ARSENAL",
"CHEMICAL SAFETY  HAZARD INVEST BRD",
"FA3047  802 CONS CC JBSA",
"IBC ACQUISITION SERVICES DIRECTORATE",
"ALASKA REGION",
"DEFENSE SUPPLY CENTER RICHMOND",
"U S ARMY DEPOT LETTERKENNY",
"NAVAL RESEARCH LABORATORY",
"NAVAL AIR TECHNICAL DATA & ENGIN",
"NAVAL AIR WARFARE CENTER, AIRCRA",
"TACOM ROCK ISLAND",
"NAVSUP WEAPON SYSTEMS SUPPORT MECH",
"NSWC INDIAN HEAD EOD TECH DIV",
"OAK RIDGE OFFICE (OFFICE OF SCIENCE)",
"IDAHO OPERATIONS OFFICE",
"HEADQUARTERS"],
    "awarding_agency_subtier_agency_code": ["Code1", "Code2", "Code3"],
    "awarding_agency_toptier_agency_code": ["TopCode1", "TopCode2", "TopCode3"],
    "awarding_office_code": ["OffCode1", "OffCode2", "OffCode3"],
    "awarding_sub_agency_name": ["Federal Transit Administration",
"National Archives and Records Administration",
"FEDERAL EMERGENCY MANAGEMENT AGENCY",
"GEOLOGICAL SURVEY",
"U.S. Geological Survey",
"Federal Prison System / Bureau of Prisons",
"Department of Housing and Urban Development",
"Bureau of the Fiscal Service",
"Defense Contract Management Agency",
"OFFICE OF POLICY, MANAGEMENT, AND BUDGET",
"U.S. Fish and Wildlife Service",
"Missile Defense Agency",
"U.S. FISH AND WILDLIFE SERVICE",
"Bureau of Indian Affairs and Bureau of Indian Education",
"BUREAU OF OCEAN ENERGY MANAGEMENT",
"US GEOLOGICAL SURVEY",
"Bureau of Ocean Energy Management",
"U. S. Coast Guard",
"ENERGY, DEPARTMENT OF",
"Federal Communications Commission",
"Immediate Office of the Secretary of Transportation",
"Federal Emergency Management Agency",
"Consumer Financial Protection Bureau",
"Federal Aviation Administration",
"Social Security Administration",
"U.S. Agency for Global Media",
"BUREAU OF INDIAN AFFAIRS",
"DEPT OF DEFENSE EDUCATION ACTIVITY (DODEA)",
"National Endowment for the Humanities",
"Natural Resources Conservation Service",
"Defense Information Systems Agency",
"National Endowment for the Arts",
"Federal Election Commission",
"U.S. Special Operations Command",
"Forest Service",
"Department of Defense Education Activity",
"National Gallery of Art",
"Federal Railroad Administration",
"Pretrial Services Agency",
"Department of the Air Force",
"Court Services and Offender Supervision Agency",
"Commodity Futures Trading Commission",
"Federal Highway Administration",
"Defense Threat Reduction Agency",
"Farm Service Agency",
"Department of the Navy",
"United States Chemical Safety Board",
"GAO, Except Comptroller General",
"United States Mint",
"Environmental Protection Agency",
"National Highway Traffic Safety Administration",
"Defense Logistics Agency",
"U.S. Coast Guard",
"U.S. Immigration and Customs Enforcement",
"Selective Service System",
"Department of the Army",
    "GAO, EXCEPT COMPTROLLER GENERAL",
"Consumer Product Safety Commission",
"Federal Acquisition Service",
"Defense Security Cooperation Agency",
"DEPT OF THE ARMY",
"DEPARTMENTAL OFFICES",
"DEFENSE CONTRACT MANAGEMENT AGENCY (DCMA)",
    "Departmental Offices", 
"Department of Energy"],
    "base_exercised_options": ["1000000.00", "1500000.00", "2000000.00"],
    "base_and_all_options": ["1200000.00", "1300000.00", "1400000.00"],
    "cage_code": ["CAGE1", "CAGE2", "CAGE3"],
    "commercial_item_acquisition_standards": ["Standard1", "Standard2", "Standard3"],
    "congressional_code": ["Congress1", "Congress2", "Congress3"],
    "contracting_offivers_determination_of_business_size": ["Small", "Medium", "Large"],
    "cost_or_pricing_data_code": ["CodeA", "CodeB", "CodeC"],
    "cost_or_pricing_data": ["Data1", "Data2", "Data3"],
    "country_code": ["USA", "CAN", "MEX"],
    "country_name": ["United States", "Canada", "Mexico"],
    "date_signed": ["2020-01-15", "2021-06-30", "2022-03-10"],
    "description": ["Description A", "Description B", "Description C"],
    "dod_acquisition_program_code": ["DOD1", "DOD2", "DOD3"],
    "dod_acquisition_program_description": ["Program Desc A", "Program Desc B", "Program Desc C"],
    "end_date": ["2021-01-15", "2022-06-30", "2023-03-10"],
    "extent_compete_description": ["Extent Desc A", "Extent Desc B", "Extent Desc C"],
    "extent_competed": ["E1", "E2", "E3"],
    "fair_opportunity_limited_sources": ["Opportunity1", "Opportunity2", "Opportunity3"],
    "funding_agency_office_name": ["Office A", "Office B", "Office C"],
    "funding_agency_subtier_agency_code": ["SubCode1", "SubCode2", "SubCode3"],
    "funding_agency_technomile_id": ["123", "456", "789"],
    "funding_agency_technomile_name": ["TechName A", "TechName B", "TechName C"],
    "funding_agency_toptier_agency_code": ["TopFundCode1", "TopFundCode2", "TopFundCode3"],
    "funding_office_code": ["FundOff1", "FundOff2", "FundOff3"],
    "funding_sub_agency_name": ["SubAgency F1", "SubAgency F2", "SubAgency F3"],
    "generated_unique_award_id": ["UID1", "UID2", "UID3"],
    "information_technology_commercial_item_category_code": ["ITCode1", "ITCode2", "ITCode3"],
    "labor_standards": ["Yes", "No", "Yes"],
    "last_modified_date": ["2022-01-15", "2022-06-30", "2022-12-10"],
    "location_country_code": ["USA", "USA", "USA"],
    "multi_year_contract": ["Yes", "No", "Yes"],
    "multi_year_contract_code": ["MYC1", "MYC2", "MYC3"],
    "naics": ["111111", "222222", "333333"],
    "naics_description": ["Agriculture", "Manufacturing", "Services"],
    "national_interest_action": ["Action1", "Action2", "Action3"],
    "national_interest_action_code": ["NAC1", "NAC2", "NAC3"],
    "number_of_actions": ["1", "2", "3"],
    "number_of_offers_received": ["3", "5", "2"],
    "other_than_full_and_open_competition_code": ["OTFOC1", "OTFOC2", "OTFOC3"],
    "parent_award_piid": ["PA1", "PA2", "PA3"],
    "parent_award_single_or_multiple": ["Single", "Multiple", "Single"],
    "parent_award_type": ["Type A", "Type B", "Type C"],
    "parent_award_type_code": ["TypeCode1", "TypeCode2", "TypeCode3"],
    "parent_recipient_name": ["Parent Rec A", "Parent Rec B", "Parent Rec C"],
    "performance_based_service_acquisition": ["Performance A", "Performance B", "Performance C"],
    "piid": ["PIID001", "PIID002", "PIID003"],
    "primary_place_of_performance_city_name": ["City A", "City B", "City C"],
    "primary_place_of_performance_zip_4": ["12345", "23456", "34567"],
    "product_or_service_code": ["PSC1", "PSC2", "PSC3"],
    "product_or_service_code_description": ["PSC Desc A", "PSC Desc B", "PSC Desc C"],
    "program_acronym": ["ACR1", "ACR2", "ACR3"],
    "recipient_name": ["Recipient A", "Recipient B", "Recipient C"],
    "recipient_parent_uei": ["UEI1", "UEI2", "UEI3"],
    "recipient_uei": ["UEI_A", "UEI_B", "UEI_C"],
    "recovered_materials_sustainability": ["Yes", "No", "Yes"],
    "solicitation_identifier": ["SID1", "SID2", "SID3"],
    "solicitation_procedure_description": ["Procedure Desc A", "Procedure Desc B", "Procedure Desc C"],
    "solicitation_procedures": ["Proc A", "Proc B", "Proc C"],
    "start_date": ["2020-01-15", "2021-06-30", "2022-03-10"],
    "state_code": ["CA", "TX", "NY"],
    "state_name": ["California", "Texas", "New York"],
    "subcontracting_plan": ["Plan A", "Plan B", "Plan C"],
    "total_obligation": ["250000.00", "750000.00", "1250000.00"],
    "type_description": ["PURCHASE ORDER", "DELIVERY ORDER", "DEFINITIVE CONTRACT", "BPA CALL", "BPA"],
    "type_of_contract_pricing": ["Fixed", "Cost-Plus", "Time & Materials"],
    "type_of_contract_pricing_code": ["F", "CP", "TM"],
    "type_of_set_aside": [
"EMERGING SMALL BUSINESS SET ASIDE",
"VERY SMALL BUSINESS",
"8(A) SOLE SOURCE",
"8A COMPETED",
"8(A) WITH HUB ZONE PREFERENCE",
"HUBZONE SET-ASIDE",
"NO SET ASIDE USED.",
"SMALL BUSINESS SET ASIDE - TOTAL",
"SERVICE DISABLED VETERAN OWNED SMALL BUSINESS SET-ASIDE",
"SMALL BUSINESS SET ASIDE - PARTIAL"],
    "type_of_set_aside_code": ["NONE", "SBA", "8a"]
}

# ------------------- Azure OpenAI Configuration -------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT", "https://tmopenaieastus2.openai.azure.com"),
    api_key=os.getenv("AZURE_OPENAI_KEY", "5c2f580e47e34102bc4d33e1c6b8d3be"),
    api_version=os.getenv("OPENAI_CHAT_API_VERSION", "2025-01-01-preview"),
    deployment_name=os.getenv("OPENAI_API_DEPLOYMENT_NAME", "gpt-4o-2"),
    temperature=0
)

# ------------------- Initialize Streamlit Session State -------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# ------------------- Query Tracking System -------------------
class QueryTracker:
    """
    Maintains context across multiple queries by tracking previous SQL queries,
    result counts, and entity mentions to enhance follow-up queries.
    """
    def __init__(self):
        # Stores the most recent SQL query executed
        self.last_sql_query = None
        # Tracks how many records were returned by the last query
        self.last_results_count = None
        # Stores additional contextual information about queries
        self.last_context = {}
        # Maps entity types to their counts and associated queries
        self.entity_mentions = {}
        # Stores just the WHERE clause from the last SQL query for reuse
        self.last_sql_where_clause = None
    
    def store_query_info(self, sql_query, results_count, context=None):
        """
        Stores information about an executed query and extracts the WHERE clause
        for potential reuse in follow-up queries.
        
        Args:
            sql_query: The SQL query that was executed
            results_count: Number of rows returned by the query
            context: Optional additional context to store
        """
        self.last_sql_query = sql_query
        self.last_results_count = results_count
        # Extract the WHERE clause using regex for potential reuse in follow-up queries
        where_match = re.search(r'WHERE\s+(.*?)(?:ORDER BY|GROUP BY|LIMIT|$)', sql_query, re.IGNORECASE | re.DOTALL)
        if where_match:
            self.last_sql_where_clause = where_match.group(1).strip()
        if context:
            self.last_context.update(context)
    
    def track_entity_mention(self, entity_type, count, query):
        """
        Records mentions of specific entities in responses, facilitating
        follow-up questions about those entities.
        
        Args:
            entity_type: The type of entity mentioned (e.g., "contract", "award")
            count: How many of these entities were mentioned
            query: The SQL query that produced these entities
        """
        self.entity_mentions[entity_type] = {"count": count, "query": query}

# Initialize the query tracker to maintain context across interactions
query_tracker = QueryTracker()

# ------------------- Database and Query Functions -------------------
def execute_sql_query(sql_query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the PostgreSQL database and returns results as a DataFrame.
    
    Args:
        sql_query: The SQL query to execute
        
    Returns:
        DataFrame containing query results or empty DataFrame on error
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def analyze_previous_response(response: str) -> dict:
    """
    Extracts entity counts from AI responses using regex patterns.
    Helps track what entities were discussed and their quantities.
    
    Args:
        response: The text of the AI's previous response
        
    Returns:
        Dictionary mapping entity types to their mentioned counts
    """
    # Regex patterns to match various ways entities might be counted in responses
    count_patterns = [
        r"There (?:are|were) (\d+) ([a-zA-Z\s]+)",
        r"Found (\d+) ([a-zA-Z\s]+)",
        r"Identified (\d+) ([a-zA-Z\s]+)",
        r"(\d+) ([a-zA-Z\s]+) (?:are|were) found",
        r"total of (\d+) ([a-zA-Z\s]+)"
    ]
    entities = {}
    for pattern in count_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            count, entity_type = match
            # Normalize entity types by removing plurals and extra spaces
            normalized_type = entity_type.strip().lower()
            if normalized_type.endswith('s'):
                normalized_type = normalized_type[:-1]
            entities[normalized_type] = int(count)
    return entities

def generate_sql_query(user_query: str) -> str:
    """
    Generates a SQL query from a natural language question using the LLM.
    Incorporates conversation history and previous query context.
    
    Args:
        user_query: The natural language question from the user
        
    Returns:
        SQL query string ready to execute
    """
    # Prepare context information for the AI
    schema_context = json.dumps(COLUMN_DEFINITIONS, indent=2)
    sample_context = json.dumps(SAMPLE_DATA, indent=2)
    
    # Get conversation history from memory
    chat_history = st.session_state.memory.chat_memory.messages
    previous_ai_messages = [msg.content for msg in chat_history if isinstance(msg, AIMessage)]
    
    # Extract entity mentions from the most recent AI response if available
    previous_entity_mentions = analyze_previous_response(previous_ai_messages[-1]) if previous_ai_messages else {}
    
    # Format recent conversation history for context
    chat_history_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                                   for msg in chat_history[-4:]])
    
    # Create context string for previously mentioned entities
    entity_context = "".join([f"- You previously mentioned there are {count} {entity_type}s.\n" 
                              for entity_type, count in previous_entity_mentions.items()])
    
    # Include the previous query context if available
    query_context = (f"Previous SQL query: {query_tracker.last_sql_query}\n"
                     f"Previous query result count: {query_tracker.last_results_count}\n"
                     f"Previous WHERE clause: {query_tracker.last_sql_where_clause}\n"
                     if query_tracker.last_sql_query and query_tracker.last_results_count is not None else "")
    
    # Detect if the user is asking for a list based on previous query
    list_request_patterns = [
        r"(?:list|show|give|display)(?:\s+me)?(?:\s+the)?(?:\s+all)?(?:\s+those)?(?:\s+(\d+))?(?:\s+([a-zA-Z\s]+))",
        r"what(?:\s+are)?(?:\s+those)?(?:\s+(\d+))?(?:\s+([a-zA-Z\s]+))",
        r"name(?:\s+the)?(?:\s+(\d+))?(?:\s+([a-zA-Z\s]+))"
    ]
    is_list_request = any(re.search(pattern, user_query, re.IGNORECASE) for pattern in list_request_patterns)
    
    # If this is a list request, provide special context to reuse previous WHERE clause
    list_request_context = (f"IMPORTANT: The user is asking to list entities from the previous query.\n"
                            f"Reuse WHERE clause: {query_tracker.last_sql_where_clause}\n"
                            f"Ensure {query_tracker.last_results_count} rows are returned.\n"
                            if is_list_request and query_tracker.last_sql_where_clause else "")

    # Create the prompt template for SQL generation
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert SQL analyst working with PostgreSQL."
        ),
        HumanMessagePromptTemplate.from_template(
            """
TABLE NAME: {table_name}
TABLE SCHEMA: {schema}
SAMPLE DATA: {samples}
CONVERSATION HISTORY: {chat_history}
PREVIOUSLY MENTIONED ENTITIES: {entity_context}
{query_context}
{list_request_context}
User Query: "{user_query}"

Generate ONLY the raw SQL query (no explanations or code blocks). Use ILIKE for text filters, include aggregation functions for counts/sums, and maintain previous context where applicable.
When using STRING_AGG, cast non-text columns (e.g., integers) to TEXT using ::TEXT to avoid type errors.
"""
        )
    ])
    
    # Build the LangChain pipeline to generate SQL
    chain = ({"table_name": lambda x: TABLE_NAME, "schema": lambda x: schema_context, "samples": lambda x: sample_context,
              "chat_history": lambda x: chat_history_text, "entity_context": lambda x: entity_context,
              "query_context": lambda x: query_context, "list_request_context": lambda x: list_request_context,
              "user_query": lambda x: x}
             | prompt_template | llm | StrOutputParser())
    
    # Generate the SQL query and clean up any markdown formatting
    sql_query = chain.invoke(user_query).strip().replace("```sql", "").replace("```", "")
    return sql_query

def refine_answer(user_query: str, sql_query: str, df: pd.DataFrame) -> str:
    """
    Takes raw SQL query results and generates a natural language answer.
    Formats the results appropriately based on the type of query.
    
    Args:
        user_query: Original natural language question
        sql_query: SQL query that was executed
        df: DataFrame containing the query results
        
    Returns:
        Natural language answer based on query results
    """
    # Get recent conversation history for context
    chat_history_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                                   for msg in st.session_state.memory.chat_memory.messages[-4:]])
    
    # Store information about this query execution
    query_tracker.store_query_info(sql_query, len(df))
    
    # Create the prompt template for answer generation
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an expert data analyst providing clear answers based on database query results."
        ),
        HumanMessagePromptTemplate.from_template(
            """
USER QUESTION: "{user_query}"
CONVERSATION HISTORY: {chat_history}
SQL QUERY EXECUTED: {sql_query}
QUERY RESULTS: {data_summary}
TOTAL RECORD COUNT: {record_count}

Guidelines:
1. Answer directly using the data provided.
2. Detect if the user is asking for specific contract details (e.g., 'details of contract', 'list contracts', 'show contract info'). If so:
   - Show up to the first 5 records with these fields: recipient_name, recipient_uei, naics, naics_description, awarding_agency_name.
   - Format each record clearly (e.g., '1. Recipient: [name], UEI: [uei], NAICS: [naics] - [description], Agency: [agency]').
   - Mention the total count and note full results availability.
3. For no results, explain in business terms (e.g., 'No contracts match this criteria, possibly due to...').
4. Format numbers with commas and $ for currency.
5. Don't show SQL unless asked.
6. If query seems off, suggest corrections.
7. Confirm counts when listing previously mentioned entities.
8. Include this text at the end based on record count:
   - If 0 records: 'No results to display.'
   - If 1-20 records: 'Full results can be viewed in the table below.'
   - If >20 records: 'Full results can be viewed in the table below or downloaded as a CSV.'
"""
        )
    ])
    
    # Prepare data summary for the prompt
    if df.empty:
        data_summary = "No results found."
        record_count = 0
    else:
        total_records = len(df)
        # Only use first 5 records for answer generation to keep responses concise
        preview_df = df.head(5)  # Take first 5 rows for the answer
        data_summary = (f"Showing first {min(5, total_records)} records:\n"
                        f"{preview_df.to_string(index=False)}")
        record_count = total_records
    
    # Build the LangChain pipeline to generate the answer
    chain = ({"user_query": lambda x: x[0], "sql_query": lambda x: x[1], "data_summary": lambda x: x[2], 
              "chat_history": lambda x: x[3], "record_count": lambda x: x[4]}
             | prompt_template | llm | StrOutputParser())
    
    # Generate the answer
    answer = chain.invoke((user_query, sql_query, data_summary, chat_history_text, record_count)).strip()
    
    # Extract and track any entity mentions in the generated answer
    entities = analyze_previous_response(answer)
    for entity_type, count in entities.items():
        query_tracker.track_entity_mention(entity_type, count, sql_query)
        
    return answer

# ------------------- Streamlit Interface -------------------
def main():
    """
    Main function that sets up the Streamlit interface and handles user interactions.
    Creates a chat-like interface for querying the database using natural language.
    """
    # Set up the application title and description
    st.title("GovSearch AI")
    st.markdown("Ask questions about government contracts and get insights from the database.")
    
    # Initialize session state for maintaining chat history between reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Create input form for user queries
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input("Enter your query (e.g., 'List all active task orders from Department of Defense')")
        submit_button = st.form_submit_button(label="Submit")
    
    # Process the query when the form is submitted
    if submit_button and user_query:
        with st.spinner("Processing your query..."):
            try:
                # Generate SQL query from natural language
                sql_query = generate_sql_query(user_query)
                
                # Execute the generated SQL query against the database
                df_results = execute_sql_query(sql_query)
                
                # Refine the raw results into a user-friendly answer
                refined_answer = refine_answer(user_query, sql_query, df_results)
                
                # Update chat history in the session state
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": refined_answer})
                
                # Store conversation in LangChain memory for context retention
                st.session_state.memory.chat_memory.add_user_message(user_query)
                st.session_state.memory.chat_memory.add_ai_message(refined_answer)
                
                # Display the current response section
                st.subheader("Current Response")
                st.write(f"**Generated SQL Query:**")
                st.code(sql_query, language="sql")
                st.write(f"**Answer:** {refined_answer}")
                
                # Show full results in an expandable section if results exist
                if len(df_results) > 0:
                    with st.expander("View Full Results"):
                        st.dataframe(df_results)
                
                # Provide CSV download option for large result sets
                if not df_results.empty and len(df_results) > 20:
                    # Generate timestamp for unique filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"query_results_{timestamp}.csv"
                    csv = df_results.to_csv(index=False).encode('utf-8')  # Full DataFrame for CSV
                    st.download_button(
                        label="Download Full Results as CSV",
                        data=csv,
                        file_name=csv_filename,
                        mime='text/csv'
                    )
            
            except Exception as e:
                # Handle errors and display them to the user
                st.error(f"Error: {str(e)}")
    
    # Display previous conversation history
    st.subheader("Chat History")
    # Show all messages except the last two (which are shown in Current Response)
    for msg in st.session_state.chat_history[:-2] if len(st.session_state.chat_history) > 2 else []:
        st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()