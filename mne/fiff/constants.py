# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

class Bunch(dict):
    """ Container object for datasets: dictionnary-like object that
        exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

FIFF = Bunch()

#
# Blocks
#
FIFF.FIFFB_MEAS               = 100
FIFF.FIFFB_MEAS_INFO          = 101
FIFF.FIFFB_RAW_DATA           = 102
FIFF.FIFFB_PROCESSED_DATA     = 103
FIFF.FIFFB_CONTINUOUS_DATA    = 112
FIFF.FIFFB_EVOKED             = 104
FIFF.FIFFB_ASPECT             = 105
FIFF.FIFFB_SUBJECT            = 106
FIFF.FIFFB_ISOTRAK            = 107
FIFF.FIFFB_HPI_MEAS           = 108
FIFF.FIFFB_HPI_RESULT         = 109
FIFF.FIFFB_HPI_COIL           = 110
FIFF.FIFFB_PROJECT            = 111
FIFF.FIFFB_CONTINUOUS_DATA    = 112
FIFF.FIFFB_VOID               = 114
FIFF.FIFFB_EVENTS             = 115
FIFF.FIFFB_INDEX              = 116
FIFF.FIFFB_DACQ_PARS          = 117
FIFF.FIFFB_REF                = 118
FIFF.FIFFB_SMSH_RAW_DATA      = 119
FIFF.FIFFB_SMSH_ASPECT        = 120
FIFF.FIFFB_HPI_SUBSYSTEM      = 121
FIFF.FIFFB_EPOCHS             = 122
FIFF.FIFFB_ICA                = 123

FIFF.FIFFB_PROJ               = 313
FIFF.FIFFB_PROJ_ITEM          = 314
FIFF.FIFFB_MRI                = 200
FIFF.FIFFB_MRI_SET            = 201
FIFF.FIFFB_MRI_SLICE          = 202
FIFF.FIFFB_PROCESSING_HISTORY = 900
FIFF.FIFFB_SSS_INFO           = 502
FIFF.FIFFB_SSS_CAL_ADJUST     = 503
FIFF.FIFFB_SSS_ST_INFO        = 504
FIFF.FIFFB_SSS_BASES          = 505
#
# Of general interest
#
FIFF.FIFF_FILE_ID         = 100
FIFF.FIFF_DIR_POINTER     = 101
FIFF.FIFF_BLOCK_ID        = 103
FIFF.FIFF_BLOCK_START     = 104
FIFF.FIFF_BLOCK_END       = 105
FIFF.FIFF_FREE_LIST       = 106
FIFF.FIFF_FREE_BLOCK      = 107
FIFF.FIFF_NOP             = 108
FIFF.FIFF_PARENT_FILE_ID  = 109
FIFF.FIFF_PARENT_BLOCK_ID = 110
#
#  Megacq saves the parameters in these tags
#
FIFF.FIFF_DACQ_PARS      = 150
FIFF.FIFF_DACQ_STIM      = 151

FIFF.FIFF_SFREQ       = 201
FIFF.FIFF_NCHAN       = 200
FIFF.FIFF_DATA_PACK   = 202
FIFF.FIFF_CH_INFO     = 203
FIFF.FIFF_MEAS_DATE   = 204
FIFF.FIFF_SUBJECT     = 205
FIFF.FIFF_COMMENT     = 206
FIFF.FIFF_NAVE        = 207
FIFF.FIFF_DIG_POINT   = 213
FIFF.FIFF_LOWPASS     = 219
FIFF.FIFF_COORD_TRANS = 222
FIFF.FIFF_HIGHPASS    = 223
FIFF.FIFF_NAME        = 233
FIFF.FIFF_DESCRIPTION = FIFF.FIFF_COMMENT
#
# Pointers
#
FIFF.FIFFV_NEXT_SEQ    = 0
FIFF.FIFFV_NEXT_NONE   = -1
#
# Channel types
#
FIFF.FIFFV_MEG_CH     =   1
FIFF.FIFFV_REF_MEG_CH = 301
FIFF.FIFFV_EEG_CH     =   2
FIFF.FIFFV_MCG_CH     = 201
FIFF.FIFFV_STIM_CH    =   3
FIFF.FIFFV_EOG_CH     = 202
FIFF.FIFFV_EMG_CH     = 302
FIFF.FIFFV_ECG_CH     = 402
FIFF.FIFFV_MISC_CH    = 502
FIFF.FIFFV_RESP_CH    = 602                # Respiration monitoring
#
# Quaternion channels for head position monitoring
#
FIFF.FIFFV_QUAT_0   = 700   # Quaternion param q0 obsolete for unit quaternion
FIFF.FIFFV_QUAT_1   = 701   # Quaternion param q1 rotation
FIFF.FIFFV_QUAT_2   = 702   # Quaternion param q2 rotation
FIFF.FIFFV_QUAT_3   = 703   # Quaternion param q3 rotation
FIFF.FIFFV_QUAT_4   = 704   # Quaternion param q4 translation
FIFF.FIFFV_QUAT_5   = 705   # Quaternion param q5 translation
FIFF.FIFFV_QUAT_6   = 706   # Quaternion param q6 translation
FIFF.FIFFV_HPI_G    = 707   # Goodness-of-fit in continuous hpi
FIFF.FIFFV_HPI_ERR  = 708   # Estimation error in continuous hpi
FIFF.FIFFV_HPI_MOV  = 709   # Estimated head movement speed in continuous hpi
#
# Coordinate frames
#
FIFF.FIFFV_COORD_UNKNOWN        = 0
FIFF.FIFFV_COORD_DEVICE         = 1
FIFF.FIFFV_COORD_ISOTRAK        = 2
FIFF.FIFFV_COORD_HPI            = 3
FIFF.FIFFV_COORD_HEAD           = 4
FIFF.FIFFV_COORD_MRI            = 5
FIFF.FIFFV_COORD_MRI_SLICE      = 6
FIFF.FIFFV_COORD_MRI_DISPLAY    = 7
FIFF.FIFFV_COORD_DICOM_DEVICE   = 8
FIFF.FIFFV_COORD_IMAGING_DEVICE = 9
#
# Needed for raw and evoked-response data
#
FIFF.FIFF_FIRST_SAMPLE   = 208
FIFF.FIFF_LAST_SAMPLE    = 209
FIFF.FIFF_ASPECT_KIND    = 210
FIFF.FIFF_DATA_BUFFER    = 300    # Buffer containing measurement data
FIFF.FIFF_DATA_SKIP      = 301    # Data skip in buffers
FIFF.FIFF_EPOCH          = 302    # Buffer containing one epoch and channel
FIFF.FIFF_DATA_SKIP_SAMP = 303    # Data skip in samples
FIFF.FIFF_MNE_BASELINE_MIN   = 304    # Time of baseline beginning
FIFF.FIFF_MNE_BASELINE_MAX   = 305    # Time of baseline end
#
# Info on subject
#
FIFF.FIFF_SUBJ_ID           = 400  # Subject ID
FIFF.FIFF_SUBJ_FIRST_NAME   = 401  # First name of the subject
FIFF.FIFF_SUBJ_MIDDLE_NAME  = 402  # Middle name of the subject
FIFF.FIFF_SUBJ_LAST_NAME    = 403  # Last name of the subject
FIFF.FIFF_SUBJ_BIRTH_DAY    = 404  # Birthday of the subject
FIFF.FIFF_SUBJ_SEX          = 405  # Sex of the subject
FIFF.FIFF_SUBJ_HAND         = 406  # Handedness of the subject
FIFF.FIFF_SUBJ_WEIGHT       = 407  # Weight of the subject
FIFF.FIFF_SUBJ_HEIGHT       = 408  # Height of the subject
FIFF.FIFF_SUBJ_COMMENT      = 409  # Comment about the subject
FIFF.FIFF_SUBJ_HIS_ID       = 410  # ID used in the Hospital Information System
#
# Different aspects of data
#
FIFF.FIFFV_ASPECT_AVERAGE       = 100  # Normal average of epochs
FIFF.FIFFV_ASPECT_STD_ERR       = 101  # Std. error of mean
FIFF.FIFFV_ASPECT_SINGLE        = 102  # Single epoch cut out from the continuous data
FIFF.FIFFV_ASPECT_SUBAVERAGE    = 103
FIFF.FIFFV_ASPECT_ALTAVERAGE    = 104  # Alternating subaverage
FIFF.FIFFV_ASPECT_SAMPLE        = 105  # A sample cut out by graph
FIFF.FIFFV_ASPECT_POWER_DENSITY = 106  # Power density spectrum
FIFF.FIFFV_ASPECT_DIPOLE_WAVE   = 200  # Dipole amplitude curve
#
# BEM surface IDs
#
FIFF.FIFFV_BEM_SURF_ID_UNKNOWN    = -1
FIFF.FIFFV_BEM_SURF_ID_BRAIN      = 1
FIFF.FIFFV_BEM_SURF_ID_SKULL      = 3
FIFF.FIFFV_BEM_SURF_ID_HEAD       = 4
#
# More of those defined in MNE
#
FIFF.FIFFV_MNE_SURF_UNKNOWN       = -1
FIFF.FIFFV_MNE_SURF_LEFT_HEMI     = 101
FIFF.FIFFV_MNE_SURF_RIGHT_HEMI    = 102
#
#   These relate to the Isotrak data
#
FIFF.FIFFV_POINT_CARDINAL = 1
FIFF.FIFFV_POINT_HPI      = 2
FIFF.FIFFV_POINT_EEG      = 3
FIFF.FIFFV_POINT_ECG      = FIFF.FIFFV_POINT_EEG
FIFF.FIFFV_POINT_EXTRA    = 4

FIFF.FIFFV_POINT_LPA = 1
FIFF.FIFFV_POINT_NASION = 2
FIFF.FIFFV_POINT_RPA = 3
#
#   SSP
#
FIFF.FIFF_PROJ_ITEM_KIND         = 3411
FIFF.FIFF_PROJ_ITEM_TIME         = 3412
FIFF.FIFF_PROJ_ITEM_NVEC         = 3414
FIFF.FIFF_PROJ_ITEM_VECTORS      = 3415
FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST = 3417
#
#   MRIs
#
FIFF.FIFF_MRI_SOURCE_PATH       = 1101
FIFF.FIFF_MRI_SOURCE_FORMAT     = 2002
FIFF.FIFF_MRI_PIXEL_ENCODING    = 2003
FIFF.FIFF_MRI_PIXEL_DATA_OFFSET = 2004
FIFF.FIFF_MRI_PIXEL_SCALE       = 2005
FIFF.FIFF_MRI_PIXEL_DATA        = 2006
FIFF.FIFF_MRI_WIDTH             = 2010
FIFF.FIFF_MRI_WIDTH_M           = 2011
FIFF.FIFF_MRI_HEIGHT            = 2012
FIFF.FIFF_MRI_HEIGHT_M          = 2013
FIFF.FIFF_MRI_DEPTH             = 2014
FIFF.FIFF_MRI_DEPTH_M           = 2015
#
FIFF.FIFFV_MRI_PIXEL_BYTE       = 1
FIFF.FIFFV_MRI_PIXEL_WORD       = 2
FIFF.FIFFV_MRI_PIXEL_SWAP_WORD  = 3
FIFF.FIFFV_MRI_PIXEL_FLOAT      = 4
#
#   These are the MNE fiff definitions
#
FIFF.FIFFB_MNE                    = 350
FIFF.FIFFB_MNE_SOURCE_SPACE       = 351
FIFF.FIFFB_MNE_FORWARD_SOLUTION   = 352
FIFF.FIFFB_MNE_PARENT_MRI_FILE    = 353
FIFF.FIFFB_MNE_PARENT_MEAS_FILE   = 354
FIFF.FIFFB_MNE_COV                = 355
FIFF.FIFFB_MNE_INVERSE_SOLUTION   = 356
FIFF.FIFFB_MNE_NAMED_MATRIX       = 357
FIFF.FIFFB_MNE_ENV                = 358
FIFF.FIFFB_MNE_BAD_CHANNELS       = 359
FIFF.FIFFB_MNE_VERTEX_MAP         = 360
FIFF.FIFFB_MNE_EVENTS             = 361
FIFF.FIFFB_MNE_MORPH_MAP          = 362
#
# CTF compensation data
#
FIFF.FIFFB_MNE_CTF_COMP           = 370
FIFF.FIFFB_MNE_CTF_COMP_DATA      = 371
#
# Fiff tags associated with MNE computations (3500...)
#
#
# 3500... Bookkeeping
#
FIFF.FIFF_MNE_ROW_NAMES              = 3502
FIFF.FIFF_MNE_COL_NAMES              = 3503
FIFF.FIFF_MNE_NROW                   = 3504
FIFF.FIFF_MNE_NCOL                   = 3505
FIFF.FIFF_MNE_COORD_FRAME            = 3506  # Coordinate frame employed. Defaults:
                          #  FIFFB_MNE_SOURCE_SPACE       FIFFV_COORD_MRI
                          #  FIFFB_MNE_FORWARD_SOLUTION   FIFFV_COORD_HEAD
                          #  FIFFB_MNE_INVERSE_SOLUTION   FIFFV_COORD_HEAD
FIFF.FIFF_MNE_CH_NAME_LIST           = 3507
FIFF.FIFF_MNE_FILE_NAME              = 3508  # This removes the collision with fiff_file.h (used to be 3501)
#
# 3510... 3590... Source space or surface
#
FIFF.FIFF_MNE_SOURCE_SPACE_POINTS        = 3510  # The vertices
FIFF.FIFF_MNE_SOURCE_SPACE_NORMALS       = 3511  # The vertex normals
FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS       = 3512  # How many vertices
FIFF.FIFF_MNE_SOURCE_SPACE_SELECTION     = 3513  # Which are selected to the source space
FIFF.FIFF_MNE_SOURCE_SPACE_NUSE          = 3514  # How many are in use
FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST       = 3515  # Nearest source space vertex for all vertices
FIFF.FIFF_MNE_SOURCE_SPACE_NEAREST_DIST  = 3516  # Distance to the Nearest source space vertex for all vertices
FIFF.FIFF_MNE_SOURCE_SPACE_ID            = 3517  # Identifier
FIFF.FIFF_MNE_SOURCE_SPACE_TYPE          = 3518  # Surface or volume

FIFF.FIFF_MNE_SOURCE_SPACE_VOXEL_DIMS    = 3596  # Voxel space dimensions in a volume source space
FIFF.FIFF_MNE_SOURCE_SPACE_INTERPOLATOR  = 3597  # Matrix to interpolate a volume source space into a mri volume
FIFF.FIFF_MNE_SOURCE_SPACE_MRI_FILE      = 3598  # MRI file used in the interpolation

FIFF.FIFF_MNE_SOURCE_SPACE_NTRI          = 3590  # Number of triangles
FIFF.FIFF_MNE_SOURCE_SPACE_TRIANGLES     = 3591  # The triangulation
FIFF.FIFF_MNE_SOURCE_SPACE_NUSE_TRI      = 3592  # Number of triangles corresponding to the number of vertices in use
FIFF.FIFF_MNE_SOURCE_SPACE_USE_TRIANGLES = 3593  # The triangulation of the used vertices in the source space

FIFF.FIFF_MNE_SOURCE_SPACE_DIST          = 3599  # Distances between vertices in use (along the surface)
FIFF.FIFF_MNE_SOURCE_SPACE_DIST_LIMIT    = 3600  # If distance is above this limit (in the volume) it has not been calculated
#
# 3520... Forward solution
#
FIFF.FIFF_MNE_FORWARD_SOLUTION       = 3520
FIFF.FIFF_MNE_SOURCE_ORIENTATION     = 3521  # Fixed or free
FIFF.FIFF_MNE_INCLUDED_METHODS       = 3522
FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD  = 3523
#
# 3530... Covariance matrix
#
FIFF.FIFF_MNE_COV_KIND               = 3530  # What kind of a covariance matrix
FIFF.FIFF_MNE_COV_DIM                = 3531  # Matrix dimension
FIFF.FIFF_MNE_COV                    = 3532  # Full matrix in packed representation (lower triangle)
FIFF.FIFF_MNE_COV_DIAG               = 3533  # Diagonal matrix
FIFF.FIFF_MNE_COV_EIGENVALUES        = 3534  # Eigenvalues and eigenvectors of the above
FIFF.FIFF_MNE_COV_EIGENVECTORS       = 3535
FIFF.FIFF_MNE_COV_NFREE              = 3536  # Number of degrees of freedom
#
# 3540... Inverse operator
#
# We store the inverse operator as the eigenleads, eigenfields,
# and weights
#
FIFF.FIFF_MNE_INVERSE_LEADS              = 3540   # The eigenleads
FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED     = 3546   # The eigenleads (already weighted with R^0.5)
FIFF.FIFF_MNE_INVERSE_FIELDS             = 3541   # The eigenfields
FIFF.FIFF_MNE_INVERSE_SING               = 3542   # The singular values
FIFF.FIFF_MNE_PRIORS_USED                = 3543   # Which kind of priors have been used for the source covariance matrix
FIFF.FIFF_MNE_INVERSE_FULL               = 3544   # Inverse operator as one matrix
                               # This matrix includes the whitening operator as well
                           # The regularization is applied
FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS = 3545  # Contains the orientation of one source per row
                           # The source orientations must be expressed in the coordinate system
                           # given by FIFF_MNE_COORD_FRAME
#
# 3550... Saved environment info
#
FIFF.FIFF_MNE_ENV_WORKING_DIR        = 3550     # Working directory where the file was created
FIFF.FIFF_MNE_ENV_COMMAND_LINE       = 3551     # The command used to create the file
#
# 3560... Miscellaneous
#
FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE       = 3560     # Is this projection item active?
FIFF.FIFF_MNE_EVENT_LIST             = 3561     # An event list (for STI 014)
FIFF.FIFF_MNE_HEMI                   = 3562     # Hemisphere association for general purposes
#
# 3570... Morphing maps
#
FIFF.FIFF_MNE_MORPH_MAP              = 3570     # Mapping of closest vertices on the sphere
FIFF.FIFF_MNE_MORPH_MAP_FROM         = 3571     # Which subject is this map from
FIFF.FIFF_MNE_MORPH_MAP_TO           = 3572     # Which subject is this map to
#
# 3580... CTF compensation data
#
FIFF.FIFF_MNE_CTF_COMP_KIND         = 3580     # What kind of compensation
FIFF.FIFF_MNE_CTF_COMP_DATA         = 3581     # The compensation data itself
FIFF.FIFF_MNE_CTF_COMP_CALIBRATED   = 3582     # Are the coefficients calibrated?
#
# 3601... values associated with ICA decomposition
#
FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS  = 3601     # ICA interface parameters
FIFF.FIFF_MNE_ICA_CHANNEL_NAMES     = 3602     # ICA channel names
FIFF.FIFF_MNE_ICA_WHITENER          = 3603     # ICA whitener
FIFF.FIFF_MNE_ICA_PCA_PARAMS        = 3604     # _PCA parameters
FIFF.FIFF_MNE_ICA_PCA_COMPONENTS    = 3605     # _PCA components
FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR = 3606     # _PCA explained variance
FIFF.FIFF_MNE_ICA_PCA_MEAN          = 3607     # _PCA mean
FIFF.FIFF_MNE_ICA_PARAMS            = 3608     # _ICA parameters
FIFF.FIFF_MNE_ICA_COMPONENTS        = 3609     # _ICA components matrix
#
# Fiff values associated with MNE computations
#
FIFF.FIFFV_MNE_FIXED_ORI            = 1
FIFF.FIFFV_MNE_FREE_ORI             = 2

FIFF.FIFFV_MNE_MEG                  = 1
FIFF.FIFFV_MNE_EEG                  = 2
FIFF.FIFFV_MNE_MEG_EEG              = 3

FIFF.FIFFV_MNE_UNKNOWN_COV          = 0
FIFF.FIFFV_MNE_SENSOR_COV           = 1
FIFF.FIFFV_MNE_NOISE_COV            = 1         # This is what it should have been called
FIFF.FIFFV_MNE_SOURCE_COV           = 2
FIFF.FIFFV_MNE_FMRI_PRIOR_COV       = 3
FIFF.FIFFV_MNE_SIGNAL_COV           = 4         # This will be potentially employed in beamformers
FIFF.FIFFV_MNE_DEPTH_PRIOR_COV      = 5         # The depth weighting prior
FIFF.FIFFV_MNE_ORIENT_PRIOR_COV     = 6         # The orientation prior
#
# Source space types (values of FIFF_MNE_SOURCE_SPACE_TYPE)
#
FIFF.FIFFV_MNE_SPACE_UNKNOWN  = -1
FIFF.FIFFV_MNE_SPACE_SURFACE  = 1
FIFF.FIFFV_MNE_SPACE_VOLUME   = 2
FIFF.FIFFV_MNE_SPACE_DISCRETE = 3
#
# Covariance matrix channel classification
#
FIFF.FIFFV_MNE_COV_CH_UNKNOWN  = -1  # No idea
FIFF.FIFFV_MNE_COV_CH_MEG_MAG  =  0  # Axial gradiometer or magnetometer [T]
FIFF.FIFFV_MNE_COV_CH_MEG_GRAD =  1  # Planar gradiometer [T/m]
FIFF.FIFFV_MNE_COV_CH_EEG      =  2  # EEG [V]
#
# Projection item kinds
#
FIFF.FIFFV_PROJ_ITEM_NONE           = 0
FIFF.FIFFV_PROJ_ITEM_FIELD          = 1
FIFF.FIFFV_PROJ_ITEM_DIP_FIX        = 2
FIFF.FIFFV_PROJ_ITEM_DIP_ROT        = 3
FIFF.FIFFV_PROJ_ITEM_HOMOG_GRAD     = 4
FIFF.FIFFV_PROJ_ITEM_HOMOG_FIELD    = 5
FIFF.FIFFV_MNE_PROJ_ITEM_EEG_AVREF  = 10
#
# Additional coordinate frames
#
FIFF.FIFFV_MNE_COORD_TUFTS_EEG   =  300         # For Tufts EEG data
FIFF.FIFFV_MNE_COORD_CTF_DEVICE  = 1001         # CTF device coordinates
FIFF.FIFFV_MNE_COORD_CTF_HEAD    = 1004         # CTF head coordinates
FIFF.FIFFV_MNE_COORD_MRI_VOXEL   = 2001         # The MRI voxel coordinates
FIFF.FIFFV_MNE_COORD_RAS         = 2002         # Surface RAS coordinates with non-zero origin
FIFF.FIFFV_MNE_COORD_MNI_TAL     = 2003         # MNI Talairach coordinates
FIFF.FIFFV_MNE_COORD_FS_TAL_GTZ  = 2004         # FreeSurfer Talairach coordinates (MNI z > 0)
FIFF.FIFFV_MNE_COORD_FS_TAL_LTZ  = 2005         # FreeSurfer Talairach coordinates (MNI z < 0)
FIFF.FIFFV_MNE_COORD_FS_TAL      = 2006         # FreeSurfer Talairach coordinates
#
# CTF coil and channel types
#
FIFF.FIFFV_REF_MEG_CH             = 301
#
#   Data types
#
FIFF.FIFFT_VOID                  = 0
FIFF.FIFFT_BYTE                  = 1
FIFF.FIFFT_SHORT                 = 2
FIFF.FIFFT_INT                   = 3
FIFF.FIFFT_FLOAT                 = 4
FIFF.FIFFT_DOUBLE                = 5
FIFF.FIFFT_JULIAN                = 6
FIFF.FIFFT_USHORT                = 7
FIFF.FIFFT_UINT                  = 8
FIFF.FIFFT_ULONG                 = 9
FIFF.FIFFT_STRING                = 10
FIFF.FIFFT_LONG                  = 11
FIFF.FIFFT_DAU_PACK13            = 13
FIFF.FIFFT_DAU_PACK14            = 14
FIFF.FIFFT_DAU_PACK16            = 16
FIFF.FIFFT_COMPLEX_FLOAT         = 20
FIFF.FIFFT_COMPLEX_DOUBLE        = 21
FIFF.FIFFT_OLD_PACK              = 23
FIFF.FIFFT_CH_INFO_STRUCT        = 30
FIFF.FIFFT_ID_STRUCT             = 31
FIFF.FIFFT_DIR_ENTRY_STRUCT      = 32
FIFF.FIFFT_DIG_POINT_STRUCT      = 33
FIFF.FIFFT_CH_POS_STRUCT         = 34
FIFF.FIFFT_COORD_TRANS_STRUCT    = 35
FIFF.FIFFT_DIG_STRING_STRUCT     = 36
FIFF.FIFFT_STREAM_SEGMENT_STRUCT = 37
#
# Units of measurement
#
FIFF.FIFF_UNIT_NONE = -1
#
# SI base units
#
FIFF.FIFF_UNIT_M   = 1
FIFF.FIFF_UNIT_KG  = 2
FIFF.FIFF_UNIT_SEC = 3
FIFF.FIFF_UNIT_A   = 4
FIFF.FIFF_UNIT_K   = 5
FIFF.FIFF_UNIT_MOL = 6
#
# SI Supplementary units
#
FIFF.FIFF_UNIT_RAD = 7
FIFF.FIFF_UNIT_SR  = 8
#
# SI base candela
#
FIFF.FIFF_UNIT_CD  = 9
#
# SI derived units
#
FIFF.FIFF_UNIT_HZ  = 101
FIFF.FIFF_UNIT_N   = 102
FIFF.FIFF_UNIT_PA  = 103
FIFF.FIFF_UNIT_J   = 104
FIFF.FIFF_UNIT_W   = 105
FIFF.FIFF_UNIT_C   = 106
FIFF.FIFF_UNIT_V   = 107
FIFF.FIFF_UNIT_F   = 108
FIFF.FIFF_UNIT_OHM = 109
FIFF.FIFF_UNIT_MHO = 110
FIFF.FIFF_UNIT_WB  = 111
FIFF.FIFF_UNIT_T   = 112
FIFF.FIFF_UNIT_H   = 113
FIFF.FIFF_UNIT_CEL = 114
FIFF.FIFF_UNIT_LM  = 115
FIFF.FIFF_UNIT_LX  = 116
#
# Others we need
#
FIFF.FIFF_UNIT_T_M   = 201  # T/m
FIFF.FIFF_UNIT_AM    = 202  # Am
FIFF.FIFF_UNIT_AM_M2 = 203  # Am/m^2
FIFF.FIFF_UNIT_AM_M3 = 204  # Am/m^3
#
# Multipliers
#
FIFF.FIFF_UNITM_E    = 18
FIFF.FIFF_UNITM_PET  = 15
FIFF.FIFF_UNITM_T    = 12
FIFF.FIFF_UNITM_MEG  = 6
FIFF.FIFF_UNITM_K    = 3
FIFF.FIFF_UNITM_H    = 2
FIFF.FIFF_UNITM_DA   = 1
FIFF.FIFF_UNITM_NONE = 0
FIFF.FIFF_UNITM_D    = -1
FIFF.FIFF_UNITM_C    = -2
FIFF.FIFF_UNITM_M    = -3
FIFF.FIFF_UNITM_MU   = -6
FIFF.FIFF_UNITM_N    = -9
FIFF.FIFF_UNITM_P    = -12
FIFF.FIFF_UNITM_F    = -15
FIFF.FIFF_UNITM_A    = -18
#
# Digitization point details
#
FIFF.FIFFV_POINT_CARDINAL = 1
FIFF.FIFFV_POINT_HPI      = 2
FIFF.FIFFV_POINT_EEG      = 3
FIFF.FIFFV_POINT_ECG      = FIFF.FIFFV_POINT_EEG
FIFF.FIFFV_POINT_EXTRA    = 4

FIFF.FIFFV_POINT_LPA      = 1
FIFF.FIFFV_POINT_NASION   = 2
FIFF.FIFFV_POINT_RPA      = 3
#
# Coil types
#
FIFF.FIFFV_COIL_NONE                  = 0  # The location info contains no data
FIFF.FIFFV_COIL_EEG                   = 1  # EEG electrode position in r0
FIFF.FIFFV_COIL_NM_122                = 2  # Neuromag 122 coils
FIFF.FIFFV_COIL_NM_24                 = 3  # Old 24 channel system in HUT
FIFF.FIFFV_COIL_NM_MCG_AXIAL          = 4  # The axial devices in the HUCS MCG system
FIFF.FIFFV_COIL_EEG_BIPOLAR           = 5  # Bipolar EEG lead

FIFF.FIFFV_COIL_DIPOLE             = 200  # Time-varying dipole definition
# The coil info contains dipole location (r0) and
# direction (ex)
FIFF.FIFFV_COIL_MCG_42             = 1000  # For testing the MCG software

FIFF.FIFFV_COIL_POINT_MAGNETOMETER = 2000  # Simple point magnetometer
FIFF.FIFFV_COIL_AXIAL_GRAD_5CM     = 2001  # Generic axial gradiometer

FIFF.FIFFV_COIL_VV_PLANAR_W        = 3011  # VV prototype wirewound planar sensor
FIFF.FIFFV_COIL_VV_PLANAR_T1       = 3012  # Vectorview SQ20483N planar gradiometer
FIFF.FIFFV_COIL_VV_PLANAR_T2       = 3013  # Vectorview SQ20483N-A planar gradiometer
FIFF.FIFFV_COIL_VV_PLANAR_T3       = 3014  # Vectorview SQ20950N planar gradiometer
FIFF.FIFFV_COIL_VV_MAG_W           = 3021  # VV prototype wirewound magnetometer
FIFF.FIFFV_COIL_VV_MAG_T1          = 3022  # Vectorview SQ20483N magnetometer
FIFF.FIFFV_COIL_VV_MAG_T2          = 3023  # Vectorview SQ20483-A magnetometer
FIFF.FIFFV_COIL_VV_MAG_T3          = 3024  # Vectorview SQ20950N magnetometer

FIFF.FIFFV_COIL_MAGNES_MAG         = 4001  # Magnes WH magnetometer
FIFF.FIFFV_COIL_MAGNES_GRAD        = 4002  # Magnes WH gradiometer
FIFF.FIFFV_COIL_CTF_GRAD           = 5001  # CTF axial gradiometer
