# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ..utils._bunch import BunchConstNamed

FIFF = BunchConstNamed()

#
# FIFF version number in use
#
FIFF.FIFFC_MAJOR_VERSION = 1
FIFF.FIFFC_MINOR_VERSION = 4
FIFF.FIFFC_VERSION = FIFF.FIFFC_MAJOR_VERSION << 16 | FIFF.FIFFC_MINOR_VERSION

#
# Blocks
#
FIFF.FIFFB_ROOT = 999
FIFF.FIFFB_MEAS = 100
FIFF.FIFFB_MEAS_INFO = 101
FIFF.FIFFB_RAW_DATA = 102
FIFF.FIFFB_PROCESSED_DATA = 103
FIFF.FIFFB_EVOKED = 104
FIFF.FIFFB_ASPECT = 105
FIFF.FIFFB_SUBJECT = 106
FIFF.FIFFB_ISOTRAK = 107
FIFF.FIFFB_HPI_MEAS = 108  # HPI measurement
FIFF.FIFFB_HPI_RESULT = 109  # Result of a HPI fitting procedure
FIFF.FIFFB_HPI_COIL = 110  # Data acquired from one HPI coil
FIFF.FIFFB_PROJECT = 111
FIFF.FIFFB_CONTINUOUS_DATA = 112
FIFF.FIFFB_CH_INFO = 113  # Extra channel information
FIFF.FIFFB_VOID = 114
FIFF.FIFFB_EVENTS = 115
FIFF.FIFFB_INDEX = 116
FIFF.FIFFB_DACQ_PARS = 117
FIFF.FIFFB_REF = 118
FIFF.FIFFB_IAS_RAW_DATA = 119
FIFF.FIFFB_IAS_ASPECT = 120
FIFF.FIFFB_HPI_SUBSYSTEM = 121
# FIFF.FIFFB_PHANTOM_SUBSYSTEM  = 122
# FIFF.FIFFB_STATUS_SUBSYSTEM   = 123
FIFF.FIFFB_DEVICE = 124
FIFF.FIFFB_HELIUM = 125
FIFF.FIFFB_CHANNEL_INFO = 126

FIFF.FIFFB_SPHERE = 300  # Concentric sphere model related
FIFF.FIFFB_BEM = 310  # Boundary-element method
FIFF.FIFFB_BEM_SURF = 311  # Boundary-element method surfaces
FIFF.FIFFB_CONDUCTOR_MODEL = 312  # One conductor model definition
FIFF.FIFFB_PROJ = 313
FIFF.FIFFB_PROJ_ITEM = 314
FIFF.FIFFB_MRI = 200
FIFF.FIFFB_MRI_SET = 201
FIFF.FIFFB_MRI_SLICE = 202
FIFF.FIFFB_MRI_SCENERY = 203  # These are for writing unrelated 'slices'
FIFF.FIFFB_MRI_SCENE = 204  # Which are actually 3D scenes...
FIFF.FIFFB_MRI_SEG = 205  # MRI segmentation data
FIFF.FIFFB_MRI_SEG_REGION = 206  # One MRI segmentation region
FIFF.FIFFB_PROCESSING_HISTORY = 900
FIFF.FIFFB_PROCESSING_RECORD = 901

FIFF.FIFFB_DATA_CORRECTION = 500
FIFF.FIFFB_CHANNEL_DECOUPLER = 501
FIFF.FIFFB_SSS_INFO = 502
FIFF.FIFFB_SSS_CAL = 503
FIFF.FIFFB_SSS_ST_INFO = 504
FIFF.FIFFB_SSS_BASES = 505
FIFF.FIFFB_IAS = 510
#
# Of general interest
#
FIFF.FIFF_FILE_ID = 100
FIFF.FIFF_DIR_POINTER = 101
FIFF.FIFF_BLOCK_ID = 103
FIFF.FIFF_BLOCK_START = 104
FIFF.FIFF_BLOCK_END = 105
FIFF.FIFF_FREE_LIST = 106
FIFF.FIFF_FREE_BLOCK = 107
FIFF.FIFF_NOP = 108
FIFF.FIFF_PARENT_FILE_ID = 109
FIFF.FIFF_PARENT_BLOCK_ID = 110
FIFF.FIFF_BLOCK_NAME = 111
FIFF.FIFF_BLOCK_VERSION = 112
FIFF.FIFF_CREATOR = 113  # Program that created the file (string)
FIFF.FIFF_MODIFIER = 114  # Program that modified the file (string)
FIFF.FIFF_REF_ROLE = 115
FIFF.FIFF_REF_FILE_ID = 116
FIFF.FIFF_REF_FILE_NUM = 117
FIFF.FIFF_REF_FILE_NAME = 118
#
#  Megacq saves the parameters in these tags
#
FIFF.FIFF_DACQ_PARS = 150
FIFF.FIFF_DACQ_STIM = 151

FIFF.FIFF_DEVICE_TYPE = 152
FIFF.FIFF_DEVICE_MODEL = 153
FIFF.FIFF_DEVICE_SERIAL = 154
FIFF.FIFF_DEVICE_SITE = 155

FIFF.FIFF_HE_LEVEL_RAW = 156
FIFF.FIFF_HELIUM_LEVEL = 157
FIFF.FIFF_ORIG_FILE_GUID = 158
FIFF.FIFF_UTC_OFFSET = 159

FIFF.FIFF_NCHAN = 200
FIFF.FIFF_SFREQ = 201
FIFF.FIFF_DATA_PACK = 202
FIFF.FIFF_CH_INFO = 203
FIFF.FIFF_MEAS_DATE = 204
FIFF.FIFF_SUBJECT = 205
FIFF.FIFF_COMMENT = 206
FIFF.FIFF_NAVE = 207
FIFF.FIFF_FIRST_SAMPLE = 208  # The first sample of an epoch
FIFF.FIFF_LAST_SAMPLE = 209  # The last sample of an epoch
FIFF.FIFF_ASPECT_KIND = 210
FIFF.FIFF_REF_EVENT = 211
FIFF.FIFF_EXPERIMENTER = 212
FIFF.FIFF_DIG_POINT = 213
FIFF.FIFF_CH_POS = 214
FIFF.FIFF_HPI_SLOPES = 215  # HPI data
FIFF.FIFF_HPI_NCOIL = 216
FIFF.FIFF_REQ_EVENT = 217
FIFF.FIFF_REQ_LIMIT = 218
FIFF.FIFF_LOWPASS = 219
FIFF.FIFF_BAD_CHS = 220
FIFF.FIFF_ARTEF_REMOVAL = 221
FIFF.FIFF_COORD_TRANS = 222
FIFF.FIFF_HIGHPASS = 223
FIFF.FIFF_CH_CALS = 224  # This will not occur in new files
FIFF.FIFF_HPI_BAD_CHS = 225  # List of channels considered to be bad in hpi
FIFF.FIFF_HPI_CORR_COEFF = 226  # HPI curve fit correlations
FIFF.FIFF_EVENT_COMMENT = 227  # Comment about the events used in averaging
FIFF.FIFF_NO_SAMPLES = 228  # Number of samples in an epoch
FIFF.FIFF_FIRST_TIME = 229  # Time scale minimum

FIFF.FIFF_SUBAVE_SIZE = 230  # Size of a subaverage
FIFF.FIFF_SUBAVE_FIRST = 231  # The first epoch # contained in the subaverage
FIFF.FIFF_NAME = 233  # Intended to be a short name.
FIFF.FIFF_DESCRIPTION = FIFF.FIFF_COMMENT  # (Textual) Description of an object
FIFF.FIFF_DIG_STRING = 234  # String of digitized points
FIFF.FIFF_LINE_FREQ = 235  # Line frequency
FIFF.FIFF_GANTRY_ANGLE = 282  # Tilt angle of the gantry in degrees.

#
# HPI fitting program tags
#
FIFF.FIFF_HPI_COIL_FREQ = 236  # HPI coil excitation frequency
FIFF.FIFF_HPI_COIL_MOMENTS = (
    240  # Estimated moment vectors for the HPI coil magnetic dipoles
)
FIFF.FIFF_HPI_FIT_GOODNESS = 241  # Three floats indicating the goodness of fit
FIFF.FIFF_HPI_FIT_ACCEPT = 242  # Bitmask indicating acceptance (see below)
FIFF.FIFF_HPI_FIT_GOOD_LIMIT = 243  # Limit for the goodness-of-fit
FIFF.FIFF_HPI_FIT_DIST_LIMIT = 244  # Limit for the coil distance difference
FIFF.FIFF_HPI_COIL_NO = 245  # Coil number listed by HPI measurement
FIFF.FIFF_HPI_COILS_USED = (
    246  # List of coils finally used when the transformation was computed
)
FIFF.FIFF_HPI_DIGITIZATION_ORDER = (
    247  # Which Isotrak digitization point corresponds to each of the coils energized
)


#
# Tags used for storing channel info
#
FIFF.FIFF_CH_SCAN_NO = (
    250  # Channel scan number. Corresponds to fiffChInfoRec.scanNo field
)
FIFF.FIFF_CH_LOGICAL_NO = (
    251  # Channel logical number. Corresponds to fiffChInfoRec.logNo field
)
FIFF.FIFF_CH_KIND = 252  # Channel type. Corresponds to fiffChInfoRec.kind field"
FIFF.FIFF_CH_RANGE = (
    253  # Conversion from recorded number to (possibly virtual) voltage at the output"
)
FIFF.FIFF_CH_CAL = 254  # Calibration coefficient from output voltage to some real units
FIFF.FIFF_CH_LOC = 255  # Channel loc
FIFF.FIFF_CH_UNIT = 256  # Unit of the data
FIFF.FIFF_CH_UNIT_MUL = 257  # Unit multiplier exponent
FIFF.FIFF_CH_DACQ_NAME = 258  # Name of the channel in the data acquisition system. Corresponds to fiffChInfoRec.name.
FIFF.FIFF_CH_COIL_TYPE = 350  # Coil type in coil_def.dat
FIFF.FIFF_CH_COORD_FRAME = 351  # Coordinate frame (integer)

#
# Pointers
#
FIFF.FIFFV_NEXT_SEQ = 0
FIFF.FIFFV_NEXT_NONE = -1
#
# Channel types
#
FIFF.FIFFV_BIO_CH = 102
FIFF.FIFFV_MEG_CH = 1
FIFF.FIFFV_REF_MEG_CH = 301
FIFF.FIFFV_EEG_CH = 2
FIFF.FIFFV_MCG_CH = 201
FIFF.FIFFV_STIM_CH = 3
FIFF.FIFFV_EOG_CH = 202
FIFF.FIFFV_EMG_CH = 302
FIFF.FIFFV_ECG_CH = 402
FIFF.FIFFV_MISC_CH = 502
FIFF.FIFFV_RESP_CH = 602  # Respiration monitoring
FIFF.FIFFV_SEEG_CH = 802  # stereotactic EEG
FIFF.FIFFV_DBS_CH = 803  # deep brain stimulation
FIFF.FIFFV_SYST_CH = 900  # some system status information (on Triux systems only)
FIFF.FIFFV_ECOG_CH = 902
FIFF.FIFFV_IAS_CH = 910  # Internal Active Shielding data (maybe on Triux only)
FIFF.FIFFV_EXCI_CH = 920  # flux excitation channel used to be a stimulus channel
FIFF.FIFFV_DIPOLE_WAVE = 1000  # Dipole time curve (xplotter/xfit)
FIFF.FIFFV_GOODNESS_FIT = 1001  # Goodness of fit (xplotter/xfit)
FIFF.FIFFV_FNIRS_CH = 1100  # Functional near-infrared spectroscopy
FIFF.FIFFV_TEMPERATURE_CH = 1200  # Functional near-infrared spectroscopy
FIFF.FIFFV_GALVANIC_CH = 1300  # Galvanic skin response
FIFF.FIFFV_EYETRACK_CH = 1400  # Eye-tracking

_ch_kind_named = {
    key: key
    for key in (
        FIFF.FIFFV_BIO_CH,
        FIFF.FIFFV_MEG_CH,
        FIFF.FIFFV_REF_MEG_CH,
        FIFF.FIFFV_EEG_CH,
        FIFF.FIFFV_MCG_CH,
        FIFF.FIFFV_STIM_CH,
        FIFF.FIFFV_EOG_CH,
        FIFF.FIFFV_EMG_CH,
        FIFF.FIFFV_ECG_CH,
        FIFF.FIFFV_MISC_CH,
        FIFF.FIFFV_RESP_CH,
        FIFF.FIFFV_SEEG_CH,
        FIFF.FIFFV_DBS_CH,
        FIFF.FIFFV_SYST_CH,
        FIFF.FIFFV_ECOG_CH,
        FIFF.FIFFV_IAS_CH,
        FIFF.FIFFV_EXCI_CH,
        FIFF.FIFFV_DIPOLE_WAVE,
        FIFF.FIFFV_GOODNESS_FIT,
        FIFF.FIFFV_FNIRS_CH,
        FIFF.FIFFV_GALVANIC_CH,
        FIFF.FIFFV_TEMPERATURE_CH,
        FIFF.FIFFV_EYETRACK_CH,
    )
}

#
# Quaternion channels for head position monitoring
#
FIFF.FIFFV_QUAT_0 = 700  # Quaternion param q0 obsolete for unit quaternion
FIFF.FIFFV_QUAT_1 = 701  # Quaternion param q1 rotation
FIFF.FIFFV_QUAT_2 = 702  # Quaternion param q2 rotation
FIFF.FIFFV_QUAT_3 = 703  # Quaternion param q3 rotation
FIFF.FIFFV_QUAT_4 = 704  # Quaternion param q4 translation
FIFF.FIFFV_QUAT_5 = 705  # Quaternion param q5 translation
FIFF.FIFFV_QUAT_6 = 706  # Quaternion param q6 translation
FIFF.FIFFV_HPI_G = 707  # Goodness-of-fit in continuous hpi
FIFF.FIFFV_HPI_ERR = 708  # Estimation error in continuous hpi
FIFF.FIFFV_HPI_MOV = 709  # Estimated head movement speed in continuous hpi
#
# Coordinate frames
#
FIFF.FIFFV_COORD_UNKNOWN = 0
FIFF.FIFFV_COORD_DEVICE = 1
FIFF.FIFFV_COORD_ISOTRAK = 2
FIFF.FIFFV_COORD_HPI = 3
FIFF.FIFFV_COORD_HEAD = 4
FIFF.FIFFV_COORD_MRI = 5
FIFF.FIFFV_COORD_MRI_SLICE = 6
FIFF.FIFFV_COORD_MRI_DISPLAY = 7
FIFF.FIFFV_COORD_DICOM_DEVICE = 8
FIFF.FIFFV_COORD_IMAGING_DEVICE = 9
_coord_frame_named = {
    key: key
    for key in (
        FIFF.FIFFV_COORD_UNKNOWN,
        FIFF.FIFFV_COORD_DEVICE,
        FIFF.FIFFV_COORD_ISOTRAK,
        FIFF.FIFFV_COORD_HPI,
        FIFF.FIFFV_COORD_HEAD,
        FIFF.FIFFV_COORD_MRI,
        # We never use these but could add at some point
        # FIFF.FIFFV_COORD_MRI_SLICE,
        # FIFF.FIFFV_COORD_MRI_DISPLAY,
        # FIFF.FIFFV_COORD_DICOM_DEVICE,
        # FIFF.FIFFV_COORD_IMAGING_DEVICE,
    )
}
#
# Needed for raw and evoked-response data
#
FIFF.FIFF_DATA_BUFFER = 300  # Buffer containing measurement data
FIFF.FIFF_DATA_SKIP = 301  # Data skip in buffers
FIFF.FIFF_EPOCH = 302  # Buffer containing one epoch and channel
FIFF.FIFF_DATA_SKIP_SAMP = 303  # Data skip in samples

#
# Info on subject
#
FIFF.FIFF_SUBJ_ID = 400  # Subject ID
FIFF.FIFF_SUBJ_FIRST_NAME = 401  # First name of the subject
FIFF.FIFF_SUBJ_MIDDLE_NAME = 402  # Middle name of the subject
FIFF.FIFF_SUBJ_LAST_NAME = 403  # Last name of the subject
FIFF.FIFF_SUBJ_BIRTH_DAY = 404  # Birthday of the subject
FIFF.FIFF_SUBJ_SEX = 405  # Sex of the subject
FIFF.FIFF_SUBJ_HAND = 406  # Handedness of the subject
FIFF.FIFF_SUBJ_WEIGHT = 407  # Weight of the subject in kg
FIFF.FIFF_SUBJ_HEIGHT = 408  # Height of the subject in m
FIFF.FIFF_SUBJ_COMMENT = 409  # Comment about the subject
FIFF.FIFF_SUBJ_HIS_ID = 410  # ID used in the Hospital Information System

FIFF.FIFFV_SUBJ_HAND_RIGHT = 1  # Righthanded
FIFF.FIFFV_SUBJ_HAND_LEFT = 2  # Lefthanded
FIFF.FIFFV_SUBJ_HAND_AMBI = 3  # Ambidextrous

FIFF.FIFFV_SUBJ_SEX_UNKNOWN = 0  # Unknown gender
FIFF.FIFFV_SUBJ_SEX_MALE = 1  # Male
FIFF.FIFFV_SUBJ_SEX_FEMALE = 2  # Female

FIFF.FIFF_PROJ_ID = 500
FIFF.FIFF_PROJ_NAME = 501
FIFF.FIFF_PROJ_AIM = 502
FIFF.FIFF_PROJ_PERSONS = 503
FIFF.FIFF_PROJ_COMMENT = 504

FIFF.FIFF_EVENT_CHANNELS = 600  # Event channel numbers
FIFF.FIFF_EVENT_LIST = 601  # List of events (integers: <sample before after>
FIFF.FIFF_EVENT_CHANNEL = 602  # Event channel
FIFF.FIFF_EVENT_BITS = 603  # Event bits array

#
# Tags used in saving SQUID characteristics etc.
#
FIFF.FIFF_SQUID_BIAS = 701
FIFF.FIFF_SQUID_OFFSET = 702
FIFF.FIFF_SQUID_GATE = 703
#
# Aspect values used to save characteristic curves of SQUIDs. (mjk)
#
FIFF.FIFFV_ASPECT_IFII_LOW = 1100
FIFF.FIFFV_ASPECT_IFII_HIGH = 1101
FIFF.FIFFV_ASPECT_GATE = 1102

#
# Values for file references
#
FIFF.FIFFV_ROLE_PREV_FILE = 1
FIFF.FIFFV_ROLE_NEXT_FILE = 2

#
# References
#
FIFF.FIFF_REF_PATH = 1101

#
# Different aspects of data
#
FIFF.FIFFV_ASPECT_AVERAGE = 100  # Normal average of epochs
FIFF.FIFFV_ASPECT_STD_ERR = 101  # Std. error of mean
FIFF.FIFFV_ASPECT_SINGLE = 102  # Single epoch cut out from the continuous data
FIFF.FIFFV_ASPECT_SUBAVERAGE = 103  # Partial average (subaverage)
FIFF.FIFFV_ASPECT_ALTAVERAGE = 104  # Alternating subaverage
FIFF.FIFFV_ASPECT_SAMPLE = 105  # A sample cut out by graph
FIFF.FIFFV_ASPECT_POWER_DENSITY = 106  # Power density spectrum
FIFF.FIFFV_ASPECT_DIPOLE_WAVE = 200  # Dipole amplitude curve

#
# BEM surface IDs
#
FIFF.FIFFV_BEM_SURF_ID_UNKNOWN = -1
FIFF.FIFFV_BEM_SURF_ID_NOT_KNOWN = 0
FIFF.FIFFV_BEM_SURF_ID_BRAIN = 1
FIFF.FIFFV_BEM_SURF_ID_CSF = 2
FIFF.FIFFV_BEM_SURF_ID_SKULL = 3
FIFF.FIFFV_BEM_SURF_ID_HEAD = 4

FIFF.FIFF_SPHERE_ORIGIN = 3001
FIFF.FIFF_SPHERE_RADIUS = 3002

FIFF.FIFF_BEM_SURF_ID = 3101  # int    surface number
FIFF.FIFF_BEM_SURF_NAME = 3102  # string surface name
FIFF.FIFF_BEM_SURF_NNODE = 3103  # int    number of nodes on a surface
FIFF.FIFF_BEM_SURF_NTRI = 3104  # int     number of triangles on a surface
FIFF.FIFF_BEM_SURF_NODES = 3105  # float  surface nodes (nnode,3)
FIFF.FIFF_BEM_SURF_TRIANGLES = 3106  # int    surface triangles (ntri,3)
FIFF.FIFF_BEM_SURF_NORMALS = 3107  # float  surface node normal unit vectors

FIFF.FIFF_BEM_POT_SOLUTION = 3110  # float ** The solution matrix
FIFF.FIFF_BEM_APPROX = 3111  # int    approximation method, see below
FIFF.FIFF_BEM_COORD_FRAME = 3112  # The coordinate frame of the model
FIFF.FIFF_BEM_SIGMA = 3113  # Conductivity of a compartment
FIFF.FIFFV_BEM_APPROX_CONST = 1  # The constant potential approach
FIFF.FIFFV_BEM_APPROX_LINEAR = 2  # The linear potential approach

#
# More of those defined in MNE
#
FIFF.FIFFV_MNE_SURF_UNKNOWN = -1
FIFF.FIFFV_MNE_SURF_LEFT_HEMI = 101
FIFF.FIFFV_MNE_SURF_RIGHT_HEMI = 102
FIFF.FIFFV_MNE_SURF_MEG_HELMET = 201  # Use this irrespective of the system
#
#   These relate to the Isotrak data (enum(point))
#
FIFF.FIFFV_POINT_CARDINAL = 1
FIFF.FIFFV_POINT_HPI = 2
FIFF.FIFFV_POINT_EEG = 3
FIFF.FIFFV_POINT_ECG = FIFF.FIFFV_POINT_EEG
FIFF.FIFFV_POINT_EXTRA = 4
FIFF.FIFFV_POINT_HEAD = 5  # Point on the surface of the head
_dig_kind_named = {
    key: key
    for key in (
        FIFF.FIFFV_POINT_CARDINAL,
        FIFF.FIFFV_POINT_HPI,
        FIFF.FIFFV_POINT_EEG,
        FIFF.FIFFV_POINT_EXTRA,
        FIFF.FIFFV_POINT_HEAD,
    )
}
#
# Cardinal point types (enum(cardinal_point))
#
FIFF.FIFFV_POINT_LPA = 1
FIFF.FIFFV_POINT_NASION = 2
FIFF.FIFFV_POINT_RPA = 3
FIFF.FIFFV_POINT_INION = 4
_dig_cardinal_named = {
    key: key
    for key in (
        FIFF.FIFFV_POINT_LPA,
        FIFF.FIFFV_POINT_NASION,
        FIFF.FIFFV_POINT_RPA,
        FIFF.FIFFV_POINT_INION,
    )
}
#
#   SSP
#
FIFF.FIFF_PROJ_ITEM_KIND = 3411
FIFF.FIFF_PROJ_ITEM_TIME = 3412
FIFF.FIFF_PROJ_ITEM_NVEC = 3414
FIFF.FIFF_PROJ_ITEM_VECTORS = 3415
FIFF.FIFF_PROJ_ITEM_DEFINITION = 3416
FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST = 3417
#   XPlotter
FIFF.FIFF_XPLOTTER_LAYOUT = 3501  # string - "Xplotter layout tag"
#
#   MRIs
#
FIFF.FIFF_MRI_SOURCE_PATH = FIFF.FIFF_REF_PATH
FIFF.FIFF_MRI_SOURCE_FORMAT = 2002
FIFF.FIFF_MRI_PIXEL_ENCODING = 2003
FIFF.FIFF_MRI_PIXEL_DATA_OFFSET = 2004
FIFF.FIFF_MRI_PIXEL_SCALE = 2005
FIFF.FIFF_MRI_PIXEL_DATA = 2006
FIFF.FIFF_MRI_PIXEL_OVERLAY_ENCODING = 2007
FIFF.FIFF_MRI_PIXEL_OVERLAY_DATA = 2008
FIFF.FIFF_MRI_BOUNDING_BOX = 2009
FIFF.FIFF_MRI_WIDTH = 2010
FIFF.FIFF_MRI_WIDTH_M = 2011
FIFF.FIFF_MRI_HEIGHT = 2012
FIFF.FIFF_MRI_HEIGHT_M = 2013
FIFF.FIFF_MRI_DEPTH = 2014
FIFF.FIFF_MRI_DEPTH_M = 2015
FIFF.FIFF_MRI_THICKNESS = 2016
FIFF.FIFF_MRI_SCENE_AIM = 2017
FIFF.FIFF_MRI_ORIG_SOURCE_PATH = 2020
FIFF.FIFF_MRI_ORIG_SOURCE_FORMAT = 2021
FIFF.FIFF_MRI_ORIG_PIXEL_ENCODING = 2022
FIFF.FIFF_MRI_ORIG_PIXEL_DATA_OFFSET = 2023
FIFF.FIFF_MRI_VOXEL_DATA = 2030
FIFF.FIFF_MRI_VOXEL_ENCODING = 2031
FIFF.FIFF_MRI_MRILAB_SETUP = 2100
FIFF.FIFF_MRI_SEG_REGION_ID = 2200
#
FIFF.FIFFV_MRI_PIXEL_UNKNOWN = 0
FIFF.FIFFV_MRI_PIXEL_BYTE = 1
FIFF.FIFFV_MRI_PIXEL_WORD = 2
FIFF.FIFFV_MRI_PIXEL_SWAP_WORD = 3
FIFF.FIFFV_MRI_PIXEL_FLOAT = 4
FIFF.FIFFV_MRI_PIXEL_BYTE_INDEXED_COLOR = 5
FIFF.FIFFV_MRI_PIXEL_BYTE_RGB_COLOR = 6
FIFF.FIFFV_MRI_PIXEL_BYTE_RLE_RGB_COLOR = 7
FIFF.FIFFV_MRI_PIXEL_BIT_RLE = 8
#
#   These are the MNE fiff definitions (range 350-390 reserved for MNE)
#
FIFF.FIFFB_MNE = 350
FIFF.FIFFB_MNE_SOURCE_SPACE = 351
FIFF.FIFFB_MNE_FORWARD_SOLUTION = 352
FIFF.FIFFB_MNE_PARENT_MRI_FILE = 353
FIFF.FIFFB_MNE_PARENT_MEAS_FILE = 354
FIFF.FIFFB_MNE_COV = 355
FIFF.FIFFB_MNE_INVERSE_SOLUTION = 356
FIFF.FIFFB_MNE_NAMED_MATRIX = 357
FIFF.FIFFB_MNE_ENV = 358
FIFF.FIFFB_MNE_BAD_CHANNELS = 359
FIFF.FIFFB_MNE_VERTEX_MAP = 360
FIFF.FIFFB_MNE_EVENTS = 361
FIFF.FIFFB_MNE_MORPH_MAP = 362
FIFF.FIFFB_MNE_SURFACE_MAP = 363
FIFF.FIFFB_MNE_SURFACE_MAP_GROUP = 364

#
# CTF compensation data
#
FIFF.FIFFB_MNE_CTF_COMP = 370
FIFF.FIFFB_MNE_CTF_COMP_DATA = 371
FIFF.FIFFB_MNE_DERIVATIONS = 372

FIFF.FIFFB_MNE_EPOCHS = 373
FIFF.FIFFB_MNE_ICA = 374
#
# Fiff tags associated with MNE computations (3500...)
#
#
# 3500... Bookkeeping
#
FIFF.FIFF_MNE_ROW_NAMES = 3502
FIFF.FIFF_MNE_COL_NAMES = 3503
FIFF.FIFF_MNE_NROW = 3504
FIFF.FIFF_MNE_NCOL = 3505
FIFF.FIFF_MNE_COORD_FRAME = 3506  # Coordinate frame employed. Defaults:
#  FIFFB_MNE_SOURCE_SPACE       FIFFV_COORD_MRI
#  FIFFB_MNE_FORWARD_SOLUTION   FIFFV_COORD_HEAD
#  FIFFB_MNE_INVERSE_SOLUTION   FIFFV_COORD_HEAD
FIFF.FIFF_MNE_CH_NAME_LIST = 3507
FIFF.FIFF_MNE_FILE_NAME = (
    3508  # This removes the collision with fiff_file.h (used to be 3501)
)
#
# 3510... 3590... Source space or surface
#
FIFF.FIFF_MNE_SOURCE_SPACE_POINTS = 3510  # The vertices
FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS = 3511  # The vertex normals
FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS = 3512  # How many vertices
FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION = 3513  # Which are selected to the source space
FIFF.FIFF_MNE_SOURCE_SPACE_NUSE = 3514  # How many are in use
FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST = (
    3515  # Nearest source space vertex for all vertices
)
FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST_DIST = (
    3516  # Distance to the Nearest source space vertex for all vertices
)
FIFF.FIFF_MNE_SOURCE_SPACE_ID = 3517  # Identifier
FIFF.FIFF_MNE_SOURCE_SPACE_TYPE = 3518  # Surface or volume
FIFF.FIFF_MNE_SOURCE_SPACE_VERTICES = 3519  # List of vertices (zero based)

FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS = (
    3596  # Voxel space dimensions in a volume source space
)
FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR = (
    3597  # Matrix to interpolate a volume source space into a mri volume
)
FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE = 3598  # MRI file used in the interpolation

FIFF.FIFF_MNE_SOURCE_SPACE_NTRI = 3590  # Number of triangles
FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES = 3591  # The triangulation
FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI = (
    3592  # Number of triangles corresponding to the number of vertices in use
)
FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES = (
    3593  # The triangulation of the used vertices in the source space
)
FIFF.FIFF_MNE_SOURCE_SPACE_NNEIGHBORS = 3594  # Number of neighbors for each source space point (used for volume source spaces)
FIFF.FIFF_MNE_SOURCE_SPACE_NEIGHBORS = (
    3595  # Neighbors for each source space point (used for volume source spaces)
)

FIFF.FIFF_MNE_SOURCE_SPACE_DIST = (
    3599  # Distances between vertices in use (along the surface)
)
FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT = (
    3600  # If distance is above this limit (in the volume) it has not been calculated
)

FIFF.FIFF_MNE_SURFACE_MAP_DATA = 3610  # Surface map data
FIFF.FIFF_MNE_SURFACE_MAP_KIND = 3611  # Type of map

#
# 3520... Forward solution
#
FIFF.FIFF_MNE_FORWARD_SOLUTION = 3520
FIFF.FIFF_MNE_SOURCE_ORIENTATION = 3521  # Fixed or free
FIFF.FIFF_MNE_INCLUDED_METHODS = 3522
FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD = 3523
#
# 3530... Covariance matrix
#
FIFF.FIFF_MNE_COV_KIND = 3530  # What kind of a covariance matrix
FIFF.FIFF_MNE_COV_DIM = 3531  # Matrix dimension
FIFF.FIFF_MNE_COV = 3532  # Full matrix in packed representation (lower triangle)
FIFF.FIFF_MNE_COV_DIAG = 3533  # Diagonal matrix
FIFF.FIFF_MNE_COV_EIGENVALUES = 3534  # Eigenvalues and eigenvectors of the above
FIFF.FIFF_MNE_COV_EIGENVECTORS = 3535
FIFF.FIFF_MNE_COV_NFREE = 3536  # Number of degrees of freedom
FIFF.FIFF_MNE_COV_METHOD = 3537  # The estimator used
FIFF.FIFF_MNE_COV_SCORE = 3538  # Negative log-likelihood

#
# 3540... Inverse operator
#
# We store the inverse operator as the eigenleads, eigenfields,
# and weights
#
FIFF.FIFF_MNE_INVERSE_LEADS = 3540  # The eigenleads
FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED = (
    3546  # The eigenleads (already weighted with R^0.5)
)
FIFF.FIFF_MNE_INVERSE_FIELDS = 3541  # The eigenfields
FIFF.FIFF_MNE_INVERSE_SING = 3542  # The singular values
FIFF.FIFF_MNE_PRIORS_USED = (
    3543  # Which kind of priors have been used for the source covariance matrix
)
FIFF.FIFF_MNE_INVERSE_FULL = 3544  # Inverse operator as one matrix
# This matrix includes the whitening operator as well
# The regularization is applied
FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS = (
    3545  # Contains the orientation of one source per row
)
# The source orientations must be expressed in the coordinate system
# given by FIFF_MNE_COORD_FRAME
FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT = 3547  # Are the sources given in Am or Am/m^2 ?
#
# 3550... Saved environment info
#
FIFF.FIFF_MNE_ENV_WORKING_DIR = 3550  # Working directory where the file was created
FIFF.FIFF_MNE_ENV_COMMAND_LINE = 3551  # The command used to create the file
FIFF.FIFF_MNE_EXTERNAL_BIG_ENDIAN = (
    3552  # Reference to an external binary file (big-endian) */
)
FIFF.FIFF_MNE_EXTERNAL_LITTLE_ENDIAN = (
    3553  # Reference to an external binary file (little-endian) */
)
#
# 3560... Miscellaneous
#
FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE = 3560  # Is this projection item active?
FIFF.FIFF_MNE_EVENT_LIST = 3561  # An event list (for STI101 / STI 014)
FIFF.FIFF_MNE_HEMI = 3562  # Hemisphere association for general purposes
FIFF.FIFF_MNE_DATA_SKIP_NOP = 3563  # A data skip turned off in the raw data
FIFF.FIFF_MNE_ORIG_CH_INFO = 3564  # Channel information before any changes
FIFF.FIFF_MNE_EVENT_TRIGGER_MASK = 3565  # Mask applied to the trigger channel values
FIFF.FIFF_MNE_EVENT_COMMENTS = 3566  # Event comments merged into one long string
FIFF.FIFF_MNE_CUSTOM_REF = 3567  # Whether a custom reference was applied to the data
FIFF.FIFF_MNE_BASELINE_MIN = 3568  # Time of baseline beginning
FIFF.FIFF_MNE_BASELINE_MAX = 3569  # Time of baseline end
#
# 3570... Morphing maps
#
FIFF.FIFF_MNE_MORPH_MAP = 3570  # Mapping of closest vertices on the sphere
FIFF.FIFF_MNE_MORPH_MAP_FROM = 3571  # Which subject is this map from
FIFF.FIFF_MNE_MORPH_MAP_TO = 3572  # Which subject is this map to
#
# 3580... CTF compensation data
#
FIFF.FIFF_MNE_CTF_COMP_KIND = 3580  # What kind of compensation
FIFF.FIFF_MNE_CTF_COMP_DATA = 3581  # The compensation data itself
FIFF.FIFF_MNE_CTF_COMP_CALIBRATED = 3582  # Are the coefficients calibrated?

FIFF.FIFF_MNE_DERIVATION_DATA = (
    3585  # Used to store information about EEG and other derivations
)
#
# 3601... values associated with ICA decomposition
#
FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS = 3601  # ICA interface parameters
FIFF.FIFF_MNE_ICA_CHANNEL_NAMES = 3602  # ICA channel names
FIFF.FIFF_MNE_ICA_WHITENER = 3603  # ICA whitener
FIFF.FIFF_MNE_ICA_PCA_COMPONENTS = 3604  # PCA components
FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR = 3605  # PCA explained variance
FIFF.FIFF_MNE_ICA_PCA_MEAN = 3606  # PCA mean
FIFF.FIFF_MNE_ICA_MATRIX = 3607  # ICA unmixing matrix
FIFF.FIFF_MNE_ICA_BADS = 3608  # ICA bad sources
FIFF.FIFF_MNE_ICA_MISC_PARAMS = 3609  # ICA misc params
#
# Miscellaneous
#
FIFF.FIFF_MNE_KIT_SYSTEM_ID = 3612  # Unique ID assigned to KIT systems
#
# Maxfilter tags
#
FIFF.FIFF_SSS_FRAME = 263
FIFF.FIFF_SSS_JOB = 264
FIFF.FIFF_SSS_ORIGIN = 265
FIFF.FIFF_SSS_ORD_IN = 266
FIFF.FIFF_SSS_ORD_OUT = 267
FIFF.FIFF_SSS_NMAG = 268
FIFF.FIFF_SSS_COMPONENTS = 269
FIFF.FIFF_SSS_CAL_CHANS = 270
FIFF.FIFF_SSS_CAL_CORRS = 271
FIFF.FIFF_SSS_ST_CORR = 272
FIFF.FIFF_SSS_NFREE = 278
FIFF.FIFF_SSS_ST_LENGTH = 279
FIFF.FIFF_DECOUPLER_MATRIX = 800
#
# Fiff values associated with MNE computations
#
FIFF.FIFFV_MNE_UNKNOWN_ORI = 0
FIFF.FIFFV_MNE_FIXED_ORI = 1
FIFF.FIFFV_MNE_FREE_ORI = 2

FIFF.FIFFV_MNE_MEG = 1
FIFF.FIFFV_MNE_EEG = 2
FIFF.FIFFV_MNE_MEG_EEG = 3

FIFF.FIFFV_MNE_PRIORS_NONE = 0
FIFF.FIFFV_MNE_PRIORS_DEPTH = 1
FIFF.FIFFV_MNE_PRIORS_LORETA = 2
FIFF.FIFFV_MNE_PRIORS_SULCI = 3

FIFF.FIFFV_MNE_UNKNOWN_COV = 0
FIFF.FIFFV_MNE_SENSOR_COV = 1
FIFF.FIFFV_MNE_NOISE_COV = 1  # This is what it should have been called
FIFF.FIFFV_MNE_SOURCE_COV = 2
FIFF.FIFFV_MNE_FMRI_PRIOR_COV = 3
FIFF.FIFFV_MNE_SIGNAL_COV = 4  # This will be potentially employed in beamformers
FIFF.FIFFV_MNE_DEPTH_PRIOR_COV = 5  # The depth weighting prior
FIFF.FIFFV_MNE_ORIENT_PRIOR_COV = 6  # The orientation prior

#
# Output map types
#
FIFF.FIFFV_MNE_MAP_UNKNOWN = -1  # Unspecified
FIFF.FIFFV_MNE_MAP_SCALAR_CURRENT = 1  # Scalar current value
FIFF.FIFFV_MNE_MAP_SCALAR_CURRENT_SIZE = 2  # Absolute value of the above
FIFF.FIFFV_MNE_MAP_VECTOR_CURRENT = 3  # Current vector components
FIFF.FIFFV_MNE_MAP_VECTOR_CURRENT_SIZE = 4  # Vector current size
FIFF.FIFFV_MNE_MAP_T_STAT = 5  # Student's t statistic
FIFF.FIFFV_MNE_MAP_F_STAT = 6  # F statistic
FIFF.FIFFV_MNE_MAP_F_STAT_SQRT = 7  # Square root of the F statistic
FIFF.FIFFV_MNE_MAP_CHI2_STAT = 8  # (Approximate) chi^2 statistic
FIFF.FIFFV_MNE_MAP_CHI2_STAT_SQRT = (
    9  # Square root of the (approximate) chi^2 statistic
)
FIFF.FIFFV_MNE_MAP_SCALAR_CURRENT_NOISE = 10  # Current noise approximation (scalar)
FIFF.FIFFV_MNE_MAP_VECTOR_CURRENT_NOISE = 11  # Current noise approximation (vector)
#
# Source space types (values of FIFF_MNE_SOURCE_SPACE_TYPE)
#
FIFF.FIFFV_MNE_SPACE_UNKNOWN = -1
FIFF.FIFFV_MNE_SPACE_SURFACE = 1
FIFF.FIFFV_MNE_SPACE_VOLUME = 2
FIFF.FIFFV_MNE_SPACE_DISCRETE = 3
#
# Covariance matrix channel classification
#
FIFF.FIFFV_MNE_COV_CH_UNKNOWN = -1  # No idea
FIFF.FIFFV_MNE_COV_CH_MEG_MAG = 0  # Axial gradiometer or magnetometer [T]
FIFF.FIFFV_MNE_COV_CH_MEG_GRAD = 1  # Planar gradiometer [T/m]
FIFF.FIFFV_MNE_COV_CH_EEG = 2  # EEG [V]
#
# Projection item kinds
#
FIFF.FIFFV_PROJ_ITEM_NONE = 0
FIFF.FIFFV_PROJ_ITEM_FIELD = 1
FIFF.FIFFV_PROJ_ITEM_DIP_FIX = 2
FIFF.FIFFV_PROJ_ITEM_DIP_ROT = 3
FIFF.FIFFV_PROJ_ITEM_HOMOG_GRAD = 4
FIFF.FIFFV_PROJ_ITEM_HOMOG_FIELD = 5
FIFF.FIFFV_PROJ_ITEM_EEG_AVREF = (
    10  # Linear projection related to EEG average reference
)
FIFF.FIFFV_MNE_PROJ_ITEM_EEG_AVREF = (
    FIFF.FIFFV_PROJ_ITEM_EEG_AVREF
)  # backward compat alias
#
# Custom EEG references
#
FIFF.FIFFV_MNE_CUSTOM_REF_OFF = 0
FIFF.FIFFV_MNE_CUSTOM_REF_ON = 1
FIFF.FIFFV_MNE_CUSTOM_REF_CSD = 2
#
# SSS job options
#
FIFF.FIFFV_SSS_JOB_NOTHING = 0  # No SSS, just copy input to output
FIFF.FIFFV_SSS_JOB_CTC = 1  # No SSS, only cross-talk correction
FIFF.FIFFV_SSS_JOB_FILTER = 2  # Spatial maxwell filtering
FIFF.FIFFV_SSS_JOB_VIRT = 3  # Transform data to another sensor array
FIFF.FIFFV_SSS_JOB_HEAD_POS = 4  # Estimate head positions, no SSS
FIFF.FIFFV_SSS_JOB_MOVEC_FIT = 5  # Estimate and compensate head movement
FIFF.FIFFV_SSS_JOB_MOVEC_QUA = (
    6  # Compensate head movement from previously estimated head positions
)
FIFF.FIFFV_SSS_JOB_REC_ALL = 7  # Reconstruct inside and outside signals
FIFF.FIFFV_SSS_JOB_REC_IN = 8  # Reconstruct inside signals
FIFF.FIFFV_SSS_JOB_REC_OUT = 9  # Reconstruct outside signals
FIFF.FIFFV_SSS_JOB_ST = 10  # Spatio-temporal maxwell filtering
FIFF.FIFFV_SSS_JOB_TPROJ = 11  # Temporal projection, no SSS
FIFF.FIFFV_SSS_JOB_XSSS = 12  # Cross-validation SSS
FIFF.FIFFV_SSS_JOB_XSUB = 13  # Cross-validation subtraction, no SSS
FIFF.FIFFV_SSS_JOB_XWAV = 14  # Cross-validation noise waveforms
FIFF.FIFFV_SSS_JOB_NCOV = 15  # Noise covariance estimation
FIFF.FIFFV_SSS_JOB_SCOV = 16  # SSS sample covariance estimation
# }

#
# Additional coordinate frames
#
FIFF.FIFFV_MNE_COORD_TUFTS_EEG = 300  # For Tufts EEG data
FIFF.FIFFV_MNE_COORD_CTF_DEVICE = 1001  # CTF device coordinates
FIFF.FIFFV_MNE_COORD_CTF_HEAD = 1004  # CTF head coordinates
FIFF.FIFFV_MNE_COORD_DIGITIZER = (
    FIFF.FIFFV_COORD_ISOTRAK
)  # Original (Polhemus) digitizer coordinates
FIFF.FIFFV_MNE_COORD_SURFACE_RAS = FIFF.FIFFV_COORD_MRI  # The surface RAS coordinates
FIFF.FIFFV_MNE_COORD_MRI_VOXEL = 2001  # The MRI voxel coordinates
FIFF.FIFFV_MNE_COORD_RAS = 2002  # Surface RAS coordinates with non-zero origin
FIFF.FIFFV_MNE_COORD_MNI_TAL = 2003  # MNI Talairach coordinates
FIFF.FIFFV_MNE_COORD_FS_TAL_GTZ = 2004  # FreeSurfer Talairach coordinates (MNI z > 0)
FIFF.FIFFV_MNE_COORD_FS_TAL_LTZ = 2005  # FreeSurfer Talairach coordinates (MNI z < 0)
FIFF.FIFFV_MNE_COORD_FS_TAL = 2006  # FreeSurfer Talairach coordinates
#
# 4D and KIT use the same head coordinate system definition as CTF
#
FIFF.FIFFV_MNE_COORD_4D_HEAD = FIFF.FIFFV_MNE_COORD_CTF_HEAD
FIFF.FIFFV_MNE_COORD_KIT_HEAD = FIFF.FIFFV_MNE_COORD_CTF_HEAD
_coord_frame_named.update({
    key: key
    for key in (
        FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
        FIFF.FIFFV_MNE_COORD_MRI_VOXEL,
        FIFF.FIFFV_MNE_COORD_RAS,
        FIFF.FIFFV_MNE_COORD_MNI_TAL,
        FIFF.FIFFV_MNE_COORD_FS_TAL,
        FIFF.FIFFV_MNE_COORD_KIT_HEAD,
    )
})

#
#   FWD Types
#

FWD = BunchConstNamed()

FWD.COIL_UNKNOWN = 0
FWD.COILC_UNKNOWN = 0
FWD.COILC_EEG = 1000
FWD.COILC_MAG = 1
FWD.COILC_AXIAL_GRAD = 2
FWD.COILC_PLANAR_GRAD = 3
FWD.COILC_AXIAL_GRAD2 = 4

FWD.COIL_ACCURACY_POINT = 0
FWD.COIL_ACCURACY_NORMAL = 1
FWD.COIL_ACCURACY_ACCURATE = 2

FWD.BEM_IP_APPROACH_LIMIT = 0.1

FWD.BEM_LIN_FIELD_SIMPLE = 1
FWD.BEM_LIN_FIELD_FERGUSON = 2
FWD.BEM_LIN_FIELD_URANKAR = 3

#
#   Data types
#
FIFF.FIFFT_VOID = 0
FIFF.FIFFT_BYTE = 1
FIFF.FIFFT_SHORT = 2
FIFF.FIFFT_INT = 3
FIFF.FIFFT_FLOAT = 4
FIFF.FIFFT_DOUBLE = 5
FIFF.FIFFT_JULIAN = 6
FIFF.FIFFT_USHORT = 7
FIFF.FIFFT_UINT = 8
FIFF.FIFFT_ULONG = 9
FIFF.FIFFT_STRING = 10
FIFF.FIFFT_LONG = 11
FIFF.FIFFT_DAU_PACK13 = 13
FIFF.FIFFT_DAU_PACK14 = 14
FIFF.FIFFT_DAU_PACK16 = 16
FIFF.FIFFT_COMPLEX_FLOAT = 20
FIFF.FIFFT_COMPLEX_DOUBLE = 21
FIFF.FIFFT_OLD_PACK = 23
FIFF.FIFFT_CH_INFO_STRUCT = 30
FIFF.FIFFT_ID_STRUCT = 31
FIFF.FIFFT_DIR_ENTRY_STRUCT = 32
FIFF.FIFFT_DIG_POINT_STRUCT = 33
FIFF.FIFFT_CH_POS_STRUCT = 34
FIFF.FIFFT_COORD_TRANS_STRUCT = 35
FIFF.FIFFT_DIG_STRING_STRUCT = 36
FIFF.FIFFT_STREAM_SEGMENT_STRUCT = 37
FIFF.FIFFT_MATRIX = 0x40000000  # 1073741824, 1 << 30
FIFF.FIFFT_SPARSE_CCS_MATRIX = 0x00100000  # 1048576
FIFF.FIFFT_SPARSE_RCS_MATRIX = 0x00200000  # 2097152

#
# Units of measurement
#
FIFF.FIFF_UNIT_NONE = -1
#
# SI base units
#
FIFF.FIFF_UNIT_UNITLESS = 0
FIFF.FIFF_UNIT_M = 1  # meter
FIFF.FIFF_UNIT_KG = 2  # kilogram
FIFF.FIFF_UNIT_SEC = 3  # second
FIFF.FIFF_UNIT_A = 4  # ampere
FIFF.FIFF_UNIT_K = 5  # Kelvin
FIFF.FIFF_UNIT_MOL = 6  # mole
#
# SI Supplementary units
#
FIFF.FIFF_UNIT_RAD = 7  # radian
FIFF.FIFF_UNIT_SR = 8  # steradian
#
# SI base candela
#
FIFF.FIFF_UNIT_CD = 9  # candela
#
# SI derived units
#
FIFF.FIFF_UNIT_MOL_M3 = 10  # mol/m^3
FIFF.FIFF_UNIT_HZ = 101  # hertz
FIFF.FIFF_UNIT_N = 102  # Newton
FIFF.FIFF_UNIT_PA = 103  # pascal
FIFF.FIFF_UNIT_J = 104  # joule
FIFF.FIFF_UNIT_W = 105  # watt
FIFF.FIFF_UNIT_C = 106  # coulomb
FIFF.FIFF_UNIT_V = 107  # volt
FIFF.FIFF_UNIT_F = 108  # farad
FIFF.FIFF_UNIT_OHM = 109  # ohm
FIFF.FIFF_UNIT_S = 110  # Siemens (same as Moh, what fiff-constants calls it)
FIFF.FIFF_UNIT_WB = 111  # weber
FIFF.FIFF_UNIT_T = 112  # tesla
FIFF.FIFF_UNIT_H = 113  # Henry
FIFF.FIFF_UNIT_CEL = 114  # celsius
FIFF.FIFF_UNIT_LM = 115  # lumen
FIFF.FIFF_UNIT_LX = 116  # lux
FIFF.FIFF_UNIT_V_M2 = 117  # V/m^2
#
# Others we need
#
FIFF.FIFF_UNIT_T_M = 201  # T/m
FIFF.FIFF_UNIT_AM = 202  # Am
FIFF.FIFF_UNIT_AM_M2 = 203  # Am/m^2
FIFF.FIFF_UNIT_AM_M3 = 204  # Am/m^3

FIFF.FIFF_UNIT_PX = 210  # Pixel
_ch_unit_named = {
    key: key
    for key in (
        FIFF.FIFF_UNIT_NONE,
        FIFF.FIFF_UNIT_UNITLESS,
        FIFF.FIFF_UNIT_M,
        FIFF.FIFF_UNIT_KG,
        FIFF.FIFF_UNIT_SEC,
        FIFF.FIFF_UNIT_A,
        FIFF.FIFF_UNIT_K,
        FIFF.FIFF_UNIT_MOL,
        FIFF.FIFF_UNIT_RAD,
        FIFF.FIFF_UNIT_SR,
        FIFF.FIFF_UNIT_CD,
        FIFF.FIFF_UNIT_MOL_M3,
        FIFF.FIFF_UNIT_HZ,
        FIFF.FIFF_UNIT_N,
        FIFF.FIFF_UNIT_PA,
        FIFF.FIFF_UNIT_J,
        FIFF.FIFF_UNIT_W,
        FIFF.FIFF_UNIT_C,
        FIFF.FIFF_UNIT_V,
        FIFF.FIFF_UNIT_F,
        FIFF.FIFF_UNIT_OHM,
        FIFF.FIFF_UNIT_S,
        FIFF.FIFF_UNIT_WB,
        FIFF.FIFF_UNIT_T,
        FIFF.FIFF_UNIT_H,
        FIFF.FIFF_UNIT_CEL,
        FIFF.FIFF_UNIT_LM,
        FIFF.FIFF_UNIT_LX,
        FIFF.FIFF_UNIT_V_M2,
        FIFF.FIFF_UNIT_T_M,
        FIFF.FIFF_UNIT_AM,
        FIFF.FIFF_UNIT_AM_M2,
        FIFF.FIFF_UNIT_AM_M3,
        FIFF.FIFF_UNIT_PX,
    )
}
#
# Multipliers
#
FIFF.FIFF_UNITM_E = 18
FIFF.FIFF_UNITM_PET = 15
FIFF.FIFF_UNITM_T = 12
FIFF.FIFF_UNITM_GIG = 9
FIFF.FIFF_UNITM_MEG = 6
FIFF.FIFF_UNITM_K = 3
FIFF.FIFF_UNITM_H = 2
FIFF.FIFF_UNITM_DA = 1
FIFF.FIFF_UNITM_NONE = 0
FIFF.FIFF_UNITM_D = -1
FIFF.FIFF_UNITM_C = -2
FIFF.FIFF_UNITM_M = -3
FIFF.FIFF_UNITM_MU = -6
FIFF.FIFF_UNITM_N = -9
FIFF.FIFF_UNITM_P = -12
FIFF.FIFF_UNITM_F = -15
FIFF.FIFF_UNITM_A = -18
_ch_unit_mul_named = {
    key: key
    for key in (
        FIFF.FIFF_UNITM_E,
        FIFF.FIFF_UNITM_PET,
        FIFF.FIFF_UNITM_T,
        FIFF.FIFF_UNITM_GIG,
        FIFF.FIFF_UNITM_MEG,
        FIFF.FIFF_UNITM_K,
        FIFF.FIFF_UNITM_H,
        FIFF.FIFF_UNITM_DA,
        FIFF.FIFF_UNITM_NONE,
        FIFF.FIFF_UNITM_D,
        FIFF.FIFF_UNITM_C,
        FIFF.FIFF_UNITM_M,
        FIFF.FIFF_UNITM_MU,
        FIFF.FIFF_UNITM_N,
        FIFF.FIFF_UNITM_P,
        FIFF.FIFF_UNITM_F,
        FIFF.FIFF_UNITM_A,
    )
}

#
# Coil types
#
FIFF.FIFFV_COIL_NONE = 0  # The location info contains no data
FIFF.FIFFV_COIL_EEG = 1  # EEG electrode position in r0
FIFF.FIFFV_COIL_NM_122 = 2  # Neuromag 122 coils
FIFF.FIFFV_COIL_NM_24 = 3  # Old 24 channel system in HUT
FIFF.FIFFV_COIL_NM_MCG_AXIAL = 4  # The axial devices in the HUCS MCG system
FIFF.FIFFV_COIL_EEG_BIPOLAR = 5  # Bipolar EEG lead
FIFF.FIFFV_COIL_EEG_CSD = 6  # CSD-transformed EEG lead

FIFF.FIFFV_COIL_DIPOLE = 200  # Time-varying dipole definition
# The coil info contains dipole location (r0) and
# direction (ex)
FIFF.FIFFV_COIL_FNIRS_HBO = 300  # fNIRS oxyhemoglobin
FIFF.FIFFV_COIL_FNIRS_HBR = 301  # fNIRS deoxyhemoglobin
FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE = 302  # fNIRS continuous wave amplitude
FIFF.FIFFV_COIL_FNIRS_OD = 303  # fNIRS optical density
FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE = 304  # fNIRS frequency domain AC amplitude
FIFF.FIFFV_COIL_FNIRS_FD_PHASE = 305  # fNIRS frequency domain phase
FIFF.FIFFV_COIL_FNIRS_RAW = FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE  # old alias
FIFF.FIFFV_COIL_FNIRS_TD_GATED_AMPLITUDE = 306  # fNIRS time-domain gated amplitude
FIFF.FIFFV_COIL_FNIRS_TD_MOMENTS_AMPLITUDE = 307  # fNIRS time-domain moments amplitude

FIFF.FIFFV_COIL_EYETRACK_POS = 400  # Eye-tracking gaze position
FIFF.FIFFV_COIL_EYETRACK_PUPIL = 401  # Eye-tracking pupil size

FIFF.FIFFV_COIL_MCG_42 = 1000  # For testing the MCG software

FIFF.FIFFV_COIL_POINT_MAGNETOMETER = 2000  # Simple point magnetometer
FIFF.FIFFV_COIL_AXIAL_GRAD_5CM = 2001  # Generic axial gradiometer

FIFF.FIFFV_COIL_VV_PLANAR_W = 3011  # VV prototype wirewound planar sensor
FIFF.FIFFV_COIL_VV_PLANAR_T1 = 3012  # Vectorview SQ20483N planar gradiometer
FIFF.FIFFV_COIL_VV_PLANAR_T2 = 3013  # Vectorview SQ20483N-A planar gradiometer
FIFF.FIFFV_COIL_VV_PLANAR_T3 = 3014  # Vectorview SQ20950N planar gradiometer
FIFF.FIFFV_COIL_VV_PLANAR_T4 = 3015  # Vectorview planar gradiometer (MEG-MRI)
FIFF.FIFFV_COIL_VV_MAG_W = 3021  # VV prototype wirewound magnetometer
FIFF.FIFFV_COIL_VV_MAG_T1 = 3022  # Vectorview SQ20483N magnetometer
FIFF.FIFFV_COIL_VV_MAG_T2 = 3023  # Vectorview SQ20483-A magnetometer
FIFF.FIFFV_COIL_VV_MAG_T3 = 3024  # Vectorview SQ20950N magnetometer
FIFF.FIFFV_COIL_VV_MAG_T4 = 3025  # Vectorview magnetometer (MEG-MRI)

FIFF.FIFFV_COIL_MAGNES_MAG = 4001  # Magnes WH magnetometer
FIFF.FIFFV_COIL_MAGNES_GRAD = 4002  # Magnes WH gradiometer
#
# Magnes reference sensors
#
FIFF.FIFFV_COIL_MAGNES_REF_MAG = 4003
FIFF.FIFFV_COIL_MAGNES_REF_GRAD = 4004
FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD = 4005
FIFF.FIFFV_COIL_MAGNES_R_MAG = FIFF.FIFFV_COIL_MAGNES_REF_MAG
FIFF.FIFFV_COIL_MAGNES_R_GRAD = FIFF.FIFFV_COIL_MAGNES_REF_GRAD
FIFF.FIFFV_COIL_MAGNES_R_GRAD_OFF = FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD

#
# CTF coil and channel types
#
FIFF.FIFFV_COIL_CTF_GRAD = 5001
FIFF.FIFFV_COIL_CTF_REF_MAG = 5002
FIFF.FIFFV_COIL_CTF_REF_GRAD = 5003
FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD = 5004
#
# KIT system coil types
#
FIFF.FIFFV_COIL_KIT_GRAD = 6001
FIFF.FIFFV_COIL_KIT_REF_MAG = 6002
#
# BabySQUID sensors
#
FIFF.FIFFV_COIL_BABY_GRAD = 7001
#
# BabyMEG sensors
#
FIFF.FIFFV_COIL_BABY_MAG = 7002
FIFF.FIFFV_COIL_BABY_REF_MAG = 7003
FIFF.FIFFV_COIL_BABY_REF_MAG2 = 7004
#
# Artemis123 sensors
#
FIFF.FIFFV_COIL_ARTEMIS123_GRAD = 7501
FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG = 7502
FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD = 7503
#
# QuSpin sensors
#
FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG = 8001
FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2 = 8002
#
# FieldLine sensors
#
FIFF.FIFFV_COIL_FIELDLINE_OPM_MAG_GEN1 = 8101
#
# Kernel sensors
#
FIFF.FIFFV_COIL_KERNEL_OPM_MAG_GEN1 = 8201
#
# KRISS sensors
#
FIFF.FIFFV_COIL_KRISS_GRAD = 9001
#
# Compumedics adult/pediatric gradiometer
#
FIFF.FIFFV_COIL_COMPUMEDICS_ADULT_GRAD = 9101
FIFF.FIFFV_COIL_COMPUMEDICS_PEDIATRIC_GRAD = 9102
_ch_coil_type_named = {
    key: key
    for key in (
        FIFF.FIFFV_COIL_NONE,
        FIFF.FIFFV_COIL_EEG,
        FIFF.FIFFV_COIL_NM_122,
        FIFF.FIFFV_COIL_NM_24,
        FIFF.FIFFV_COIL_NM_MCG_AXIAL,
        FIFF.FIFFV_COIL_EEG_BIPOLAR,
        FIFF.FIFFV_COIL_EEG_CSD,
        FIFF.FIFFV_COIL_DIPOLE,
        FIFF.FIFFV_COIL_FNIRS_HBO,
        FIFF.FIFFV_COIL_FNIRS_HBR,
        FIFF.FIFFV_COIL_FNIRS_RAW,
        FIFF.FIFFV_COIL_FNIRS_OD,
        FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE,
        FIFF.FIFFV_COIL_FNIRS_FD_PHASE,
        FIFF.FIFFV_COIL_FNIRS_TD_GATED_AMPLITUDE,
        FIFF.FIFFV_COIL_FNIRS_TD_MOMENTS_AMPLITUDE,
        FIFF.FIFFV_COIL_MCG_42,
        FIFF.FIFFV_COIL_EYETRACK_POS,
        FIFF.FIFFV_COIL_EYETRACK_PUPIL,
        FIFF.FIFFV_COIL_POINT_MAGNETOMETER,
        FIFF.FIFFV_COIL_AXIAL_GRAD_5CM,
        FIFF.FIFFV_COIL_VV_PLANAR_W,
        FIFF.FIFFV_COIL_VV_PLANAR_T1,
        FIFF.FIFFV_COIL_VV_PLANAR_T2,
        FIFF.FIFFV_COIL_VV_PLANAR_T3,
        FIFF.FIFFV_COIL_VV_PLANAR_T4,
        FIFF.FIFFV_COIL_VV_MAG_W,
        FIFF.FIFFV_COIL_VV_MAG_T1,
        FIFF.FIFFV_COIL_VV_MAG_T2,
        FIFF.FIFFV_COIL_VV_MAG_T3,
        FIFF.FIFFV_COIL_VV_MAG_T4,
        FIFF.FIFFV_COIL_MAGNES_MAG,
        FIFF.FIFFV_COIL_MAGNES_GRAD,
        FIFF.FIFFV_COIL_MAGNES_REF_MAG,
        FIFF.FIFFV_COIL_MAGNES_REF_GRAD,
        FIFF.FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD,
        FIFF.FIFFV_COIL_CTF_GRAD,
        FIFF.FIFFV_COIL_CTF_REF_MAG,
        FIFF.FIFFV_COIL_CTF_REF_GRAD,
        FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD,
        FIFF.FIFFV_COIL_KIT_GRAD,
        FIFF.FIFFV_COIL_KIT_REF_MAG,
        FIFF.FIFFV_COIL_BABY_GRAD,
        FIFF.FIFFV_COIL_BABY_MAG,
        FIFF.FIFFV_COIL_BABY_REF_MAG,
        FIFF.FIFFV_COIL_BABY_REF_MAG2,
        FIFF.FIFFV_COIL_ARTEMIS123_GRAD,
        FIFF.FIFFV_COIL_ARTEMIS123_REF_MAG,
        FIFF.FIFFV_COIL_ARTEMIS123_REF_GRAD,
        FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG,
        FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2,
        FIFF.FIFFV_COIL_FIELDLINE_OPM_MAG_GEN1,
        FIFF.FIFFV_COIL_KERNEL_OPM_MAG_GEN1,
        FIFF.FIFFV_COIL_KRISS_GRAD,
        FIFF.FIFFV_COIL_COMPUMEDICS_ADULT_GRAD,
        FIFF.FIFFV_COIL_COMPUMEDICS_PEDIATRIC_GRAD,
    )
}

# MNE RealTime
FIFF.FIFF_MNE_RT_COMMAND = 3700  # realtime command
FIFF.FIFF_MNE_RT_CLIENT_ID = 3701  # realtime client

# MNE epochs bookkeeping
FIFF.FIFF_MNE_EPOCHS_SELECTION = 3800  # the epochs selection
FIFF.FIFF_MNE_EPOCHS_DROP_LOG = 3801  # the drop log
FIFF.FIFF_MNE_EPOCHS_REJECT_FLAT = 3802  # rejection and flat params
FIFF.FIFF_MNE_EPOCHS_RAW_SFREQ = 3803  # original raw sfreq

# MNE annotations
FIFF.FIFFB_MNE_ANNOTATIONS = 3810  # annotations block

# MNE Metadata Dataframes
FIFF.FIFFB_MNE_METADATA = 3811  # metadata dataframes block

# Table to match unrecognized channel location names to their known aliases
CHANNEL_LOC_ALIASES = {
    # this set of aliases are published in doi:10.1097/WNP.0000000000000316 and
    # doi:10.1016/S1388-2457(00)00527-7.
    "Cb1": "POO7",
    "Cb2": "POO8",
    "CB1": "POO7",
    "CB2": "POO8",
    "T1": "T9",
    "T2": "T10",
    "T3": "T7",
    "T4": "T8",
    "T5": "T9",
    "T6": "T10",
    "M1": "TP9",
    "M2": "TP10",
    # EGI ref chan is named VREF/Vertex Ref.
    # In the standard montages for EGI, the ref is named Cz
    "VREF": "Cz",
    "Vertex Reference": "Cz"
    # add a comment here (with doi of a published source) above any new
    # aliases, as they are added
}
