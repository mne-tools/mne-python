from ..externals.tempita import Template


raw_template = Template(u"""
<li class="{{div_klass}}" id="{{id}}">
<h4>{{caption}}</h4>
<table class="table table-hover">
    <tr>
        <th>Measurement date</th>
        {{if meas_date is not None}}
        <td>{{meas_date}}</td>
        {{else}}<td>Unknown</td>{{endif}}
    </tr>
    <tr>
        <th>Experimenter</th>
        {{if info['experimenter'] is not None}}
        <td>{{info['experimenter']}}</td>
        {{else}}<td>Unknown</td>{{endif}}
    </tr>
    <tr>
        <th>Digitized points</th>
        {{if info['dig'] is not None}}
        <td>{{len(info['dig'])}} points</td>
        {{else}}
        <td>Not available</td>
        {{endif}}
    </tr>
    <tr>
        <th>Good channels</th>
        <td>{{n_mag}} magnetometer, {{n_grad}} gradiometer,
            and {{n_eeg}} EEG channels</td>
    </tr>
    <tr>
        <th>Bad channels</th>
        {{if info['bads'] is not None}}
        <td>{{', '.join(info['bads'])}}</td>
        {{else}}<td>None</td>{{endif}}
    </tr>
    <tr>
        <th>EOG channels</th>
        <td>{{eog}}</td>
    </tr>
    <tr>
        <th>ECG channels</th>
        <td>{{ecg}}</td>
    <tr>
        <th>Measurement time range</th>
        <td>{{u'%0.2f' % tmin}} to {{u'%0.2f' % tmax}} sec.</td>
    </tr>
    <tr>
        <th>Sampling frequency</th>
        <td>{{u'%0.2f' % info['sfreq']}} Hz</td>
    </tr>
    <tr>
        <th>Highpass</th>
        <td>{{u'%0.2f' % info['highpass']}} Hz</td>
    </tr>
     <tr>
        <th>Lowpass</th>
        <td>{{u'%0.2f' % info['lowpass']}} Hz</td>
    </tr>
</table>
</li>
""")


def _repr_raw_html(self, global_id='', caption='', tmin=0, tmax=0):
    n_eeg = len(pick_types(self, meg=False, eeg=True))
    n_grad = len(pick_types(self, meg='grad'))
    n_mag = len(pick_types(self, meg='mag'))
    pick_eog = pick_types(self, meg=False, eog=True)
    if len(pick_eog) > 0:
        eog = ', '.join(np.array(self['ch_names'])[pick_eog])
    else:
        eog = 'Not available'
    pick_ecg = pick_types(self, meg=False, ecg=True)
    if len(pick_ecg) > 0:
        ecg = ', '.join(np.array(self['ch_names'])[pick_ecg])
    else:
        ecg = 'Not available'
    meas_date = self['meas_date']
    if meas_date is not None:
        meas_date = meas_date.strftime("%B %d, %Y") + ' GMT'

    return raw_template.substitute(
        div_klass='raw', id=global_id, caption=caption, info=self,
        meas_date=meas_date, n_eeg=n_eeg, n_grad=n_grad, n_mag=n_mag,
        eog=eog, ecg=ecg, tmin=tmin, tmax=tmax)