%% clear
clear all global
close all

%% init obob
addpath('/mnt/obob/obob_ownft/');

obob_init_ft;

%% set vars...
f_name = '/mnt/sinuhe/data_raw/aw_crossfrog/subject_subject/180219/19800908igdb_block01.fif';

max_duration = 3;

%% read all data...
cfg = [];
cfg.dataset = f_name;
cfg.trialdef.triallength = Inf;

cfg = ft_definetrial(cfg);

cfg.channel = 'MEG01*';
data = ft_preprocessing(cfg);

%% take less data....
cfg = [];
cfg.latency = [0 max_duration];

data = ft_selectdata(cfg, data);

%% replace data with random stuff...
data.trial{1} = randn(size(data.trial{1}));

%% remove cfg...
data = rmfield(data, 'cfg');

%% save raw data...
save('raw_v7.mat', 'data', '-v7');
save('raw_v73.mat', 'data', '-v7.3');

%% get events...
evt_orig = ft_read_event(f_name);
evt_orig = ft_filter_event(evt_orig, 'type', 'Trigger');

n_evt = 5;

clear evt;
min_sample = ceil(0.2 * data.fsample);
max_sample = size(data.trial{1}, 2) - ceil(0.2 * data.fsample);

all_evt_samples = randi([min_sample max_sample], 1, n_evt);
all_evt_samples = sort(all_evt_samples);

for idx_evt = 1:n_evt
  evt(idx_evt) = evt_orig(1);
  evt(idx_evt).value = randi([1 2]);
  evt(idx_evt).sample = all_evt_samples(idx_evt);
end %for

evt_table = struct2table(evt);
evt_table = table(evt_table.sample-1, zeros(length(evt_table.value), 1), evt_table.value);

writetable(evt_table, 'events.eve', 'FileType', 'text', 'Delimiter', 'tab', 'WriteVariableNames', false);

%% epoch data...
cfg = [];
cfg.event = evt;
cfg.hdr = data.hdr;
cfg.trialdef.eventtype = 'Trigger';
cfg.trialdef.prestim = 0.05;
cfg.trialdef.poststim = 0.05;

trl = ft_trialfun_general(cfg);
trl(:, end+1) = randi([1 2], size(trl(:, end)));

cfg = [];
cfg.trl = trl;

data_epoched = ft_redefinetrial(cfg, data);

bad_epochs = [];
for idx_trial = 1:length(data_epoched.trial)
  if any(any(isnan(data_epoched.trial{idx_trial})))
    bad_epochs(end+1) = idx_trial;
  end %if
end %for

data_epoched.trial(bad_epochs) = [];
data_epoched.time(bad_epochs) = [];
data_epoched.trialinfo(bad_epochs, :) = [];
data_epoched.sampleinfo(bad_epochs, :) = [];

%% save epoched data...
save('epoched_v7.mat', 'data_epoched', '-v7');
save('epoched_v73.mat', 'data_epoched', '-v7.3');

%% average data....
cfg = [];

data_avg = ft_timelockanalysis(cfg, data_epoched);

%% save averaged data...
save('averaged_v7.mat', 'data_avg', '-v7');
save('averaged_v73.mat', 'data_avg', '-v7.3');