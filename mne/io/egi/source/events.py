"""Code for extract the events."""

from datetime import datetime
from glob import glob
from os.path import basename, join, splitext
from xml.etree.ElementTree import parse
from numpy import fix


# shorttime = lambda x: x[:26] + x[29:32] + x[33:]


def read_mff_events(filename, header):
    """Function for extract the events.

    Parameters:
    filename = str
    header = The header array from read_mff_header
    """
    orig = {}
    for xml_file in glob(join(filename, '*.xml')):
        xml_type = splitext(basename(xml_file))[0]
        orig[xml_type] = _parse_xml(xml_file)
    xml_files = orig.keys()
    xml_events = [x for x in xml_files if x[:7] == 'Events_']
    # start_time = datetime.strptime(shorttime(orig['info'][0]['recordTime']),
    #                                '%Y-%m-%dT%H:%M:%S.%f%z')
    start_time = _ns2py_time(orig['info'][1]['recordTime'])
    markers = []
    code = []
    for xml in xml_events:
        for event in orig[xml][2:]:
            # event_start = datetime.strptime(shorttime(event['beginTime']),
            #                                 '%Y-%m-%dT%H:%M:%S.%f%z')
            event_start = _ns2py_time(event['beginTime'])
            start = (event_start - start_time).total_seconds()
            if event['code'] not in code:
                code.append(event['code'])
            marker = {'name': event['code'],
                      'start': start,
                      'start_sample': int(fix(start * header['Fs'])),
                      'end': start + float(event['duration']) / 1e9,
                      'chan': None,
                      }
            markers.append(marker)
    events_tims = dict()
    for ev in code:
        trig_samp = list(c['start_sample'] for n,
                         c in enumerate(markers) if c['name'] == ev)
        events_tims.update({ev: trig_samp})
    return events_tims, code


def _parse_xml(xml_file):
    xml = parse(xml_file)
    root = xml.getroot()
    return _xml2list(root)


def _xml2list(root):
    output = []
    for element in root:

        if element:

            if element[0].tag != element[-1].tag:
                output.append(_xml2dict(element))
            else:
                output.append(_xml2list(element))

        elif element.text:
            text = element.text.strip()
            if text:
                tag = _ns(element.tag)
                output.append({tag: text})

    return output


def _ns(s):
    """Remove namespace, but only it there is a namespace to begin with."""
    if '}' in s:
        return '}'.join(s.split('}')[1:])
    else:
        return s


def _xml2dict(root):
    """Use functions instead of Class.

    remove namespace based on
    http://stackoverflow.com/questions/2148119
    """
    output = {}
    if root.items():
        output.update(dict(root.items()))

    for element in root:
        if element:
            if len(element) == 1 or element[0].tag != element[1].tag:
                one_dict = _xml2dict(element)
            else:
                one_dict = {_ns(element[0].tag): _xml2list(element)}

            if element.items():
                one_dict.update(dict(element.items()))
            output.update({_ns(element.tag): one_dict})

        elif element.items():
            output.update({_ns(element.tag): dict(element.items())})

        else:
            output.update({_ns(element.tag): element.text})
    return output


def _ns2py_time(nstime):
    nsdate = nstime[0:10]
    nstime0 = nstime[11:26]
    nstime00 = nsdate + " " + nstime0
    pytime = datetime.strptime(nstime00, '%Y-%m-%d %H:%M:%S.%f')
    return pytime
