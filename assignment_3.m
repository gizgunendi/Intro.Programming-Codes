%% Assignment 3 - Counterbalancing and Pseudorandomization

% Pseudorandomized block orders for 60 participants in a 2x3 design:
% Factor 1: Familiarity (“familiar” vs “unfamiliar”)
% Factor 2: Emotion (“positive”, “neutral”, “negative”)

% Half of participants (30) start with "familiar" blocks.
% One third of participants (20) start with each emotion (positive / neutral / negative).
% Blocks with the same emotion appear consecutively.
% The order of emotions is randomized for each participant
%
% Output:
% A structure variable "stimlist" containing the block order for each participant.

clear; clc;

%% 1. Base conditions
familiarity = {'familiar', 'unfamiliar'};
emotions = {'positive', 'neutral', 'negative'};

n_participants = 60;
stimlist = struct();

%% 2. Loop through participants 
% I prefer to solve the problem with "for" loop instead of the alternative with "while" loop.

for i = 1:n_participants

    % Familiarity counterbalancing 
    % First 30 participants start with "familiar", next 30 with "unfamiliar"
    if i <= n_participants/2
        fam_order = {'familiar', 'unfamiliar'};
    else
        fam_order = {'unfamiliar', 'familiar'};
    end

    % Emotion counterbalancing
    % First 20 participants start with "positive"
    % Next 20 start with "neutral"
    % Last 20 start with "negative"
    if i <= 20
        first_emotion = 'positive';
    elseif i <= 40
        first_emotion = 'neutral';
    else
        first_emotion = 'negative';
    end

    % Randomized order of emotions ensuring each group starts with their assigned one
    remaining_emotions = setdiff(emotions, first_emotion, 'stable');
    shuffled_remaining = remaining_emotions(randperm(length(remaining_emotions)));
    emotion_order = [{first_emotion}, shuffled_remaining];

    % Final block list 
    block_order = {}; % temporary cell array for 6 blocks
    for e = 1:length(emotion_order)
        emotion = emotion_order{e};

        % Build emotion pair (two familiarity levels)
        pair = {[emotion '_' fam_order{1}], [emotion '_' fam_order{2}]};

        % Append the pair to the list
        block_order = [block_order, pair];
    end

    % ----- (D) Store result -----
    stimlist(i).subject = i;
    stimlist(i).order = block_order;
end

%% 3. Example output
% Display the block order for the first few participants
for i = 1:5
    fprintf('\nParticipant %d block order:\n', i);
    disp(stimlist(i).order)
end
%% 4. Summary table for all participants
participant_ids = [stimlist.subject]';
block_orders = {stimlist.order}';

% Convert to table
stim_table = table(participant_ids, block_orders, ...
    'VariableNames', {'Participant', 'BlockOrder'});

% Display first few rows
disp(stim_table(1:5,:));
disp(stim_table.BlockOrder{1}');



