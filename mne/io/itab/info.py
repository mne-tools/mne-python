"""Build measurement info
"""

# Author: Vittorio Pizzella <vittorio.pizzella@unich.it>
#
# License: BSD (3-clause)

from datetime import datetime, timezone

import numpy as np

from ...utils import logger
from ..._fiff.meas_info import _empty_info
from ..constants import FIFF
from ...annotations import Annotations

from .constants import ITAB


def _convert_time(date_str, time_str):
    """Convert date and time strings to float time"""


    for fmt in ("%d/%m/%Y", "%d-%b-%Y", "%a, %b %d, %Y"):
        
        try:
            date = datetime.strptime(date_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError(
            'Illegal date: %s.\nIf the language of the date does not '
            'correspond to your local machine\'s language try to set the '
            'locale to the language of the date string:\n'
            'locale.setlocale(locale.LC_ALL, "en_US")' % date_str)

    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            time = datetime.strptime(time_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError('Illegal time: %s' % time_str)
    # MNE-C uses mktime which uses local time, but here we instead decouple
    # conversion location from the process, and instead assume that the
    # acquisiton was in GMT. This will be wrong for most sites, but at least
    # the value we obtain here won't depend on the geographical location
    # that the file was converted.
    res = datetime(date.year, date.month, date.day,
                   time.hour, time.minute, time.second,
                   tzinfo=timezone.utc
                   )



    return res


def _mhdch_2_chs(mhd_ch):
    """Build chs list item from mhd ch list item"""
    
    ch = dict()

    unit = 1.0  # Default unit multiplier
   # Generic channel
    loc = np.zeros(12)
    ch['ch_name'] = mhd_ch['label']
    ch['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
    ch['coil_type'] = 0
    ch['range']  = 1.0
    ch['unit'] = FIFF.FIFF_UNIT_NONE
    ch['unit_mul'] = FIFF.FIFF_UNITM_NONE
    if mhd_ch['calib'] == 0:
        ch['cal'] = 1.
    else:
        ch['cal'] = float(mhd_ch['amvbit'] / mhd_ch['calib'])
    ch['loc'] = loc
    ch['scanno'] = 0
    ch['kind'] = FIFF.FIFFV_MISC_CH
    ch['logno'] = 1

   # Magnetic channel
    if mhd_ch['type'] == ITAB.ITABV_MAG_CH:
        loc[0:12] = (
                mhd_ch['pos'][0]['posx'] / 1000,  #r0
                mhd_ch['pos'][0]['posy'] / 1000,  
                mhd_ch['pos'][0]['posz'] / 1000,
                mhd_ch['pos'][0]['orix'],  #ex
                0,
                0,
                0,                         #ey
                mhd_ch['pos'][0]['oriy'],
                0,
                0,                         #ez
                0,
                mhd_ch['pos'][0]['oriz'])         
        ch['loc'] = loc
        ch['kind'] = FIFF.FIFFV_MEG_CH
        ch['coil_type'] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
        ch['logno'] = int(mhd_ch['number'])
        if mhd_ch['calib'] == 0:
            ch['cal'] = 1.
        else:
            ch['cal'] = float(mhd_ch['amvbit'] / mhd_ch['calib'])
        ch['unit'] = FIFF.FIFF_UNIT_T
        if mhd_ch['unit'] == "fT":
            ch['unit_mul'] = FIFF.FIFF_UNITM_F
            unit = 10**-15
        elif mhd_ch['unit'] == "pT":
            ch['unit_mul'] = FIFF.FIFF_UNITM_P
            unit = 10**-12
    
   # Electric channel
    if mhd_ch['type'] == ITAB.ITABV_EEG_CH:     
        ch['kind'] = FIFF.FIFFV_BIO_CH
        ch['cal'] = float(mhd_ch['amvbit'] / mhd_ch['calib'])
        ch['unit'] = FIFF.FIFF_UNIT_V
        ch['logno'] = int(mhd_ch['number'])
        if mhd_ch['unit'] == "mV":
            ch['unit_mul'] = FIFF.FIFF_UNITM_M
            unit = 10**-3
        elif mhd_ch['unit'] == "uT":
            ch['unit_mul'] = FIFF.FIFF_UNITM_MU
            unit = 10**-6

   # Other channel type
    if (mhd_ch['type'] == ITAB.ITABV_REF_EEG_CH    and     
        mhd_ch['type'] == ITAB.ITABV_REF_MAG_CH    and 
        mhd_ch['type'] == ITAB.ITABV_REF_AUX_CH    and 
        mhd_ch['type'] == ITAB.ITABV_REF_PARAM_CH  and
        mhd_ch['type'] == ITAB.ITABV_REF_DIGIT_CH  and
        mhd_ch['type'] == ITAB.ITABV_REF_FLAG_CH  ):

        ch['kind'] = FIFF.FIFFV_BIO_CH
        ch['cal'] = float(mhd_ch['amvbit'] / mhd_ch['calib'])
        ch['unit'] = FIFF.FIFF_UNIT_V
        if mhd_ch['unit'] == "mV":
            ch['unit_mul'] = FIFF.FIFF_UNITM_M
            unit = 10**-3
        elif mhd_ch['unit'] == "uT":
            ch['unit_mul'] = FIFF.FIFF_UNITM_MU
            unit = 10**-6
    
    return ch, unit
   
   
def _mhd2info(mhd):
    """Create meas info from ITAB mhd data"""

    info = _empty_info(mhd['smpfq'])
  
    info['meas_date'] = _convert_time(mhd['date'], mhd['time'])
    
    info['description'] = mhd['notes']
    
    si = dict()
    #si['id'] = mhd['id']
    si['last_name'] = mhd['last_name']
    si['first_name'] = mhd['first_name']
    si['sex'] = int(mhd['subinfo']['sex'][0])
                
    info['subject_info'] = si

    channel_names = list()
    chs = list()
    bads = list()
    units = list()
    
    
    for channel in mhd['ch']:
        channel_names.append(channel['label'])
        
        if channel['flag'] == 1:
            bads.append(channel['label'])
        
        ch, unit = _mhdch_2_chs(channel)
        chs.append(ch)
        
        units.append(unit)
    
    info['chs']  = chs
    info['ch_names'] = channel_names
    info['bads'] = bads

    if mhd['hw_low_fr'] != 0:
        info['lowpass'] = mhd['hw_low_fr']
    
    if mhd['hw_hig_fr'] != 0:
        info['highpass'] = mhd['hw_hig_fr']
         

    # Total number of channels in .raw file
    info['nchan']    = mhd['nchan']
    
    # Total number of timepoints in .raw file
    info['temp'] = dict()
    
    info['temp']['units'] = units
    info['temp']['n_samp']   = mhd['ntpdata']
    
    # Only one trial (continous acquisition)
    info['temp']['ntrials'] = 1    

    # Data start in .raw file
    info['temp']['start_data'] = mhd['start_data']
    
    # Get Polhemus digitization data (in head coordinates) 
    dig = []

    point_sequence = [
        FIFF.FIFFV_POINT_NASION,    # Nasion
        FIFF.FIFFV_POINT_RPA,       # Right pre-auricolar
        FIFF.FIFFV_POINT_LPA,       # Left pre-auricolar
    ]
        
    for i, point in enumerate(mhd['marker']):
        point_info = dict() 

        point_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD

        # Point kind
        if i < 3:       # Nasion, Right pre-auricolar, Left pre-auricolar
            point_kind = FIFF.FIFFV_POINT_CARDINAL
        elif i == 3:    # Vertex
            point_kind = FIFF.FIFFV_POINT_EXTRA
        else:           # HPI Coils
            point_kind = FIFF.FIFFV_POINT_HPI
        
        point_info['kind']  = point_kind


        # Point identity
        if i < 3:
            point_ident = point_sequence[i]
        else:
            point_ident = i + 1
        point_info['ident'] = point_ident

        point_info['r'] = (point['posx'],
                           point['posy'],
                           point['posz'])
        
        dig += [point_info]
    
    # TDB other poosible head points, check on dig points.
      
    info['dig'] = dig

    # TBD trigger handling

    event = list()
    for sample in mhd['sample']:   
        event.append([ sample['start'], 
                       sample['type'], 
                       sample['quality'] ])
    
    info['events'] = event
    
    annotations = {
        'description': [],
        'onset': [],
        'duration': [],
    }
    for sample in mhd['segment']:
        if sample['start'] != 0:
            start = sample['start'] / info['sfreq']
        else:
            start = 0.0
        duration = sample['ntptot'] / info['sfreq']
        description = sample['type']
        
        annotations['description'].append(description)
        annotations['onset'].append(start)
        annotations['duration'].append(duration)
        
        
    info['temp']['annotations'] = Annotations(
        annotations['onset'], 
        annotations['duration'], 
        annotations['description']
    )
     
    info._check_consistency()

    logger.info('Measurement info composed.')
            
    return info