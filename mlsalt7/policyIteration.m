function [v, pi] = policyIteration(model, maxit)

% initialize the value function
v = zeros(model.stateCount, 1);
v_ = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);

for i = 1:maxit,
    v = v_;
    for s = 1:model.stateCount  % policy evaluation
         temp = sum(squeeze(model.P(s,:,:))...
                .*(repmat(model.R(s,:),model.stateCount,1)...
                + repmat(model.gamma*v,1,4)),1);
         v_(s) = temp(pi(s));
    end
    for s = 1:model.stateCount  % policy imporovement
        [~,pi(s)] = max(sum(squeeze(model.P(s,:,:))...
                    .*(repmat(model.R(s,:),model.stateCount,1)...
                    + repmat(v_,1,4)),1));
    end
    
    % exit early
    if max(abs(v_ - v))<0.0001
        fprintf('Value function converged after %d itarations!\n', i)
        break;
    end
    
end

