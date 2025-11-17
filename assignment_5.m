%% Before the Assignment | Notes/Concept of the Encoding Model
% We examine whether DNN feature representations of images 
% can predict the brain's EEG responses to the same images.
%
% A deep neural network (DNN) converts each image into a numerical feature vector.
% These features represent different levels of visual information (edges, textures,
% object parts, semantic categories).
%
% The EEG signal reflects how the human brain processes the same image over time.
%
% If a linear regression model can successfully predict EEG responses from DNN features,
% this indicates that the representations learned by the DNN resemble
% the neural representations used by the visual cortex at specific time points.
%
% In other words:
% Better prediction accuracy means stronger similarity between DNN representations
% and human visual processing at that moment in time.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assignment 5 – EEG Encoding Models
% Goal:
% (1) Test the effect of different training data amounts on encoding accuracy.
% (2) Test the effect of different DNN feature amounts on encoding accuracy.

clear; clc;

%% Load the data
load("data_assignment_5.mat")

% Data format reminders:
% dnn_train: [16540 × 100] DNN features for training images
% eeg_train: [16540 × channels × time]
% dnn_test:  [200 × 100]
% eeg_test:  [200 × channels × time]
% times:     [1 × time]

[numTrials, numChannels, numTime] = size(eeg_train);


%% 1 — EFFECT OF TRAINING DATA AMOUNT

train_sizes = [250, 1000, 10000, 16540];
meanR_all = zeros(length(train_sizes), numTime);

colors = lines(length(train_sizes));   % for plotting

for idx = 1:length(train_sizes)

    N = train_sizes(idx);

    fprintf("\nTraining models using %d image conditions...\n", N);

    % Select random training samples
    train_idx = randperm(numTrials, N);
    dnn_sub = dnn_train(train_idx, :);
    eeg_sub = eeg_train(train_idx, :, :);

    % Initialize weights & intercepts
    W = zeros(size(dnn_sub, 2), numChannels, numTime);
    b = zeros(numChannels, numTime);

    totalModels = numChannels * numTime;
    modelCount = 0;

    % Train linear regression models independently per channel × time
    for ch = 1:numChannels
        for t = 1:numTime
            y = eeg_sub(:, ch, t); % EEG vector for this channel/time
            mdl = fitlm(dnn_sub, y);
            W(:, ch, t) = mdl.Coefficients.Estimate(2:end);
            b(ch, t)    = mdl.Coefficients.Estimate(1);
            modelCount = modelCount + 1;
            fprintf('\rProgress: %d / %d', modelCount, totalModels);
        end
    end

    %% Predict EEG for the 200 test images
    eeg_pred = zeros(200, numChannels, numTime);
    for ch = 1:numChannels
        for t = 1:numTime
            eeg_pred(:, ch, t) = dnn_test * W(:, ch, t) + b(ch, t);
        end
    end

    %% Compute Pearson correlations
    R = zeros(numChannels, numTime);
    for ch = 1:numChannels
        for t = 1:numTime
            real_vec = squeeze(eeg_test(:, ch, t));
            pred_vec = squeeze(eeg_pred(:, ch, t));
            R(ch, t) = corr(real_vec, pred_vec);
        end
    end

    %% Average across channels
    meanR = mean(R, 1);

    %% Seperate Figures for Each Training Amount

    figure; 
    plot(times, meanR, 'LineWidth', 2);
    xlabel("Time (ms)");
    ylabel("Mean Pearson Correlation");
    title(sprintf("Encoding Accuracy (Training N = %d)", N));
    grid on;
    set(gca, 'FontSize', 14);

    meanR_all(idx, :) = meanR;

end

% --- Plot all training sizes in a single figure ---
figure; hold on;
for idx = 1:length(train_sizes)
    plot(times, meanR_all(idx, :), 'LineWidth', 2, 'Color', colors(idx,:));
end
xlabel("Time (ms)");
ylabel("Mean Pearson Correlation");
title("Effect of Training Data Amount on Encoding Accuracy");
legend(arrayfun(@(x) sprintf('N = %d', x), train_sizes, 'UniformOutput', false));
grid on;
set(gca, 'FontSize', 14);

%% INTERPRETATION (PART 1)
% PATTERN:
% Encoding accuracy increases systematically as the training data amount increases.
% The model trained on 250 images performs worst, 1000 images slightly better,
% 10,000 images significantly better, and the full 16,540-image model shows
% the highest prediction accuracy across almost all time points.
%
% REASON:
% EEG data is noisy and high-dimensional. With small datasets,
% the regression weights are under-constrained and cannot generalize well.
% Larger training sets provide:
% more stable estimation of the linear mapping,
% better averaging of noise,
% improved ability to capture true DNN–EEG relationships.
% Thus, more training images result in more robust brain prediction.


%% PART 2 — EFFECT OF DNN FEATURE AMOUNT

feature_sizes = [25, 50, 75, 100];
meanR_all_features = zeros(length(feature_sizes), numTime);
colors2 = lines(length(feature_sizes));

for idx = 1:length(feature_sizes)

    F = feature_sizes(idx);

    fprintf("\nTraining models using %d DNN features...\n", F);

    % Select first F features
    dnnF_train = dnn_train(:, 1:F);
    dnnF_test  = dnn_test(:, 1:F);

    % Initialize weights & intercepts
    W = zeros(F, numChannels, numTime);
    b = zeros(numChannels, numTime);

    totalModels = numChannels * numTime;
    modelCount = 0;

    % Train models
    for ch = 1:numChannels
        for t = 1:numTime
            y = eeg_train(:, ch, t);
            mdl = fitlm(dnnF_train, y);
            W(:, ch, t) = mdl.Coefficients.Estimate(2:end);
            b(ch, t)    = mdl.Coefficients.Estimate(1);
            modelCount = modelCount + 1;
            fprintf('\rProgress: %d / %d', modelCount, totalModels);
        end
    end

    %% Predict EEG for test images
    eeg_pred = zeros(200, numChannels, numTime);
    for ch = 1:numChannels
        for t = 1:numTime
            eeg_pred(:, ch, t) = dnnF_test * W(:, ch, t) + b(ch, t);
        end
    end

    %% Compute Pearson correlations
    R = zeros(numChannels, numTime);
    for ch = 1:numChannels
        for t = 1:numTime
            real_vec = squeeze(eeg_test(:, ch, t));
            pred_vec = squeeze(eeg_pred(:, ch, t));
            R(ch, t) = corr(real_vec, pred_vec);
        end
    end

    %% Average across channels
    meanR = mean(R, 1);

     %% Seperate Figures for Each Feature Amount 
    figure;
    plot(times, meanR, 'LineWidth', 2);
    xlabel("Time (ms)");
    ylabel("Mean Pearson Correlation");
    title(sprintf("Encoding Accuracy (%d DNN Features)", F));
    grid on;
    set(gca, 'FontSize', 14);

    meanR_all_features(idx, :) = meanR;

end

% --- Plot all feature sizes in a single figure ---
figure; hold on;
for idx = 1:length(feature_sizes)
    plot(times, meanR_all_features(idx, :), 'LineWidth', 2, 'Color', colors2(idx,:));
end
xlabel("Time (ms)");
ylabel("Mean Pearson Correlation");
title("Effect of DNN Feature Amount on Encoding Accuracy");
legend(arrayfun(@(x) sprintf('%d features', x), feature_sizes, 'UniformOutput', false));
grid on;
set(gca, 'FontSize', 14);

%% INTERPRETATION (PART 2)
% PATTERN:
% Encoding accuracy increases as the number of DNN features increases.
% The model using only 25 features shows the lowest accuracy.
% Accuracy improves with 50 and 75 features, and is highest with 100 features.
%
% REASON:
% More DNN features contain richer and more diverse visual information.
% EEG signals reflect multiple visual stages (edges, textures, and finally objects).
% With too few DNN features, important visual dimensions are missing.
% As the feature space grows, the regression model captures more of the
% relevant visual information, improving its ability to predict EEG responses.
