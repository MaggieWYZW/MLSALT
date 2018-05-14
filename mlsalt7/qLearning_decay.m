function [v, pi, acc_R] = qLearning_decay(model, maxit, maxeps, alpha)
rand('seed',326)
% initialize the value function
Q = zeros(model.stateCount, 4);
v = zeros(model.stateCount, 1);
v_ = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
acc_R = zeros(maxeps, 1);

for i = 1:maxeps,
    % every time we reset the episode, start at the given startState
    s = model.startState;

    for j = 1:maxit,
        % PICK AN ACTION
        epsilon = 10/i;
        prob1 = rand;
        if prob1 < epsilon 
            a = datasample([1 2 3 4], 1);
        else
            [~,a] = max(Q(s,:));
        end
        
        % look for next state s_ after action a
        p = 0;
        r = rand;
        for s_ = 1:model.stateCount,
            p = p + model.P(s, s_, a);
            if r <= p,
                break;
            end
        end

        % s_ should now be the next sampled state.
        % IMPLEMENT THE UPDATE RULE FOR Q HERE.
        acc_R(i) = acc_R(i) + model.R(s,a);

        Q(s,a) = Q(s,a) + alpha*(model.R(s,a) + model.gamma*max(Q(s_,:)) - Q(s,a));
        [v_(s), pi(s)] = max(Q(s,:));
        s = s_;
        
        % SHOULD WE BREAK OUT OF THE LOOP?
        if s == model.goalState
            fprintf('%d: reached the goal state! Start Again!\n', i)
            break;
        end
    end
    
%     if max(abs(v_-v))<0.0000001
%        fprintf('value function converged after %d episodes!', i)
%        break;
%     end
    v = v_;
end
