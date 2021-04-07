from ..externals.tempita import Template


info_template = Template("""
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
        <th>Participant</th>
        {{if info['subject_info'] is not None}}
            {{if 'his_id' in info['subject_info'].keys()}}
            <td>{{info['subject_info']['his_id']}}</td>
            {{endif}}
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
""")

raw_template = Template("""
{{info_repr[:-9]}}
    <tr>
        <th>Filenames</th>
        <td>{{', '.join(filenames)}}</td>
    </tr>
    <tr>
        <th>Duration</th>
        <td>{{duration}} (HH:MM:SS)</td>
    </tr>
</table>
""")

epochs_template = Template("""
<table class="table table-hover">
    <tr>
        <th>Number of events</th>
        <td>{{len(epochs.events)}}</td>
    </tr>
    <tr>
        <th>Events</th>
        {{if events is not None}}
        <td>{{events}}</td>
        {{else}}
        <td>Not available</td>
        {{endif}}
    </tr>
    <tr>
        <th>Time range</th>
        <td>{{f'{epochs.tmin:.3f} – {epochs.tmax:.3f} sec'}}</td>
    </tr>
    <tr>
        <th>Baseline</th>
        <td>{{baseline}}</td>
    </tr>
</table>
""")
