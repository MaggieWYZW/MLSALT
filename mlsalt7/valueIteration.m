function [v, pi] = valueIteration(model, maxit)
% initialize the value function
v = zeros(model.stateCount, 1);

for i = 1:maxit,
    % initialize the policy and the new value function
    pi = ones(model.stateCount, 1); 
    v_ = zeros(model.stateCount, 1);
    % perform the Bellman update for each state
    for s = 1:model.stateCount,
        [v_(s),pi(s)] = max(sum(squeeze(model.P(s,:,:))...
                        .*(repmat(model.R(s,:),model.stateCount,1) ...
                        + repmat(model.gamma*v,1,4)),1));
    end
    % exit early
    if max(abs(v-v_)) < 0.00001,
        fprintf('value function converge after %d iterations', i);
        break;
    end
    v = v_; %update
end

