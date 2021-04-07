import uuid
from ..externals.tempita import Template
# style, section_ids=section_ids, sections=sections,

info_template = Template("""
{{style}}

<table class="table table-hover">

    <tr>
        <th style="text-align: left;">
        <label for={{section_ids[0]}}> {{sections[0]}} </label>
        </th>
    </tr>

    <tr class="{{section_ids[0]}}">
        <th>Measurement date</th>
        {{if meas_date is not None}}
        <td>{{meas_date}}</td>
        {{else}}<td>Unknown</td>{{endif}}
    </tr>
    <tr class="{{section_ids[0]}}">
        <th>Experimenter</th>
        {{if info['experimenter'] is not None}}
        <td>{{info['experimenter']}}</td>
        {{else}}<td>Unknown</td>{{endif}}
    </tr>
    <tr  class="{{section_ids[0]}}">
        <th>Participant</th>
        {{if info['subject_info'] is not None}}
            {{if 'his_id' in info['subject_info'].keys()}}
            <td>{{info['subject_info']['his_id']}}</td>
            {{endif}}
        {{else}}<td>Unknown</td>{{endif}}
    </tr>

    <tr>
        <th style="text-align: left;">
        <label for={{section_ids[1]}}> {{sections[1]}} </label>
        </th>
    </tr>

    <tr  class="{{section_ids[1]}}">
        <th>Digitized points</th>
        {{if info['dig'] is not None}}
        <td>{{len(info['dig'])}} points</td>
        {{else}}
        <td>Not available</td>
        {{endif}}
    </tr>
    <tr  class="{{section_ids[1]}}">
        <th>Good channels</th>
        <td>{{n_mag}} magnetometer, {{n_grad}} gradiometer,
            and {{n_eeg}} EEG channels</td>
    </tr>
    <tr  class="{{section_ids[1]}}">
        <th>Bad channels</th>
        {{if info['bads'] is not None}}
        <td>{{', '.join(info['bads'])}}</td>
        {{else}}<td>None</td>{{endif}}
    </tr>
    <tr  class="{{section_ids[1]}}">
        <th>EOG channels</th>
        <td>{{eog}}</td>
    </tr>
    <tr  class="{{section_ids[1]}}">
        <th>ECG channels</th>
        <td>{{ecg}}</td>
    </tr>

    <tr>
        <th style="text-align: left;">
        <label for={{section_ids[2]}}> {{sections[2]}} </label>
        </th>
    </tr>
    <tr  class="{{section_ids[2]}}">
        <th>Sampling frequency</th>
        <td>{{u'%0.2f' % info['sfreq']}} Hz</td>
    </tr>
    <tr  class="{{section_ids[2]}}">
        <th>Highpass</th>
        <td>{{u'%0.2f' % info['highpass']}} Hz</td>
    </tr>
    <tr  class="{{section_ids[2]}}">
        <th>Lowpass</th>
        <td>{{u'%0.2f' % info['lowpass']}} Hz</td>
    </tr>
    {{if filenames is not None}}
    <tr  class="{{section_ids[2]}}">
        <th>Filenames</th>
        <td>{{', '.join(filenames)}}</td>
    </tr>
    {{endif}}
    {{if duration is not None}}
    <tr  class="{{section_ids[2]}}">
        <th>Duration</th>
        <td>{{duration}} (HH:MM:SS)</td>
    </tr>
    {{endif}}
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


def _section_style(section_id):
    html = f"""#{section_id} ~ table [for="{section_id}"]::before {{
                   display: inline-block;
                   content: "►";
                   font-size: 11px;
                   width: 15px;
                   text-align: left;
                   }}

               #{section_id}:checked ~ table [for="{section_id}"]::before {{
                   content: "▼";
                   }}

               #{section_id} ~ table tr.{section_id} {{
                   visibility: collapse;
                   }}

               #{section_id}:checked ~ table tr.{section_id} {{
                   visibility: visible;
                   }}
            """
    return html


def collapsible_sections_reprt_html(sections):
    style = "<style>  label { cursor: pointer; }"
    ids_ = []
    for section in sections:
        section_id = f"section_{str(uuid.uuid4())}"
        style += _section_style(section_id)
        ids_.append(section_id)
    style += "</style>"

    for i in ids_:
        style += f"""
        <input type="checkbox" id="{i}" hidden aria-hidden="true"/>
        """

    return style, ids_
