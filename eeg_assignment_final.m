%To do a fresh start :)
clear; close all; clc;

%% 1. Loading the data
load('eeg_data_assignment_2.mat');

% To check what is in this dataset we have;
% Each dimension represents condition, channel, and time respectively.
size(eeg)        % EEG data is a 3D matrix (200 image conditions × 63 channels × 140 time points)
length(ch_names) % list of 63 EEG channel names (like 'Fz', 'Oz' etc.)
length(times)    % time points in seconds, 140 samples


%% 2. Occipital and frontal channels
%Frontal channels (F) represent anterior regions (are at the front of the head (decision, control areas), 
% and occipital channels (O) represent posterior visual areas (are at the
% back of the head (visual proessing areas).

% Channel names that have 'O' = occipital
% Channel names that have 'F' = frontal
including_o = contains(ch_names, 'O');
including_f = contains(ch_names, 'F');

% To see which the channels are
occipital_names = ch_names(including_o);
frontal_names = ch_names(including_f);

%% Mean EEG voltage at 0.1 seconds
% % The EEG data has 140 time points
% I want to find which index is closest to 0.1 seconds.
% abs(times - 0.1) gives distance between each time and 0.1 s.
% min(...) finds the smallest distance, so time_point= index closest to 0.1 s.
[~, time_point] = min(abs(times - 0.1));

% The mean EEG value for occipital and frontal channels across all image conditions at that time.
mean_occipital_voltage = mean(eeg(:, including_o, time_point), 'all');
mean_frontal_voltage = mean(eeg(:, including_f, time_point), 'all');

% To print the results
fprintf('Mean EEG at 0.1 s (Occipital): %.4f µV\n', mean_occipital_voltage);
fprintf('Mean EEG at 0.1 s (Frontal): %.4f µV\n', mean_frontal_voltage);

% I displayed the EEG values with 4 decimal places for clarity.


%% 3. Ploting the mean EEG across all image conditions and channels
% I average across the first 2 dimensions (conditions and channels)to get a single timecourse.
%Chat Gpt recommended to use squzeeze function
mean_all = squeeze(mean(mean(eeg, 1), 2)); 

% To plot it
figure;
plot(times, mean_all, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Mean EEG (µV)');
title('Mean EEG across all conditions and channels');
grid on;

%The overall mean EEG waveform shows a clear deviation around 0.1 s, 
% which could reflect an early sensory response.


%% 4. Comparing occipital vs frontal timecourses
% I want to compare those two regions.
% Again, I average across conditions and channels separately for each region.

mean_occipital = squeeze(mean(mean(eeg(:, including_o, :), 1), 2));
mean_frontal = squeeze(mean(mean(eeg(:, including_f, :), 1), 2));

% I plot both on the same figure:

figure;
plot(times, mean_occipital, 'b', 'LineWidth', 2); 

hold on;

plot(times, mean_frontal, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Mean EEG (µV)');
legend({'Occipital', 'Frontal'});
title('Occipital vs Frontal mean EEG');

grid on;

% The plot shows how the EEG voltage changes over time, averaged across everything (images and the channels). 

% If I take the difference between occipital and frontal activity (occipital - frontal),
% a positive value means the occipital region shows stronger voltage than the frontal region 
% A negative value means the frontal region becomes more active than the occipital one

% It seems the occipital activity peaks earlier (around 100 ms) and the channel might have early visual responses,
% and the frontal activity increases later (~300 ms), 
% suggesting later cognitive processing or evaluation 
% and it might be slower because of decision or control processes.



%% 5. Comparing condition 1 and condition 2 (for occipital channels)

% The first dimension of EEG is "image condition".
% I pick the first two for this example.

cond1 = squeeze(mean(eeg(1, including_o, :), 2));
cond2 = squeeze(mean(eeg(2, including_o, :), 2));

figure;
plot(times, cond1, 'b', 'LineWidth', 2); hold on;
plot(times, cond2, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('EEG (µV)');
legend({'Condition 1', 'Condition 2'});
title('Occipital Mean EEG: Image Condition 1 vs Image Condition 2');
grid on;

% If the two conditions differ, I might see amplitude or timing differences.
% I think the two lines might look a bit different depending on the image type.
% Early time (0-0.2s) = visual processing, later (0.3-0.6s) = higher meaning.

% I think early differences between 0 and 200 milliseconds usually reflect perceptual processing, 
% while later differences between 300 and 600 milliseconds are related to
% meaning, attention, planning or evaluation processes 
% (basically executive functions).