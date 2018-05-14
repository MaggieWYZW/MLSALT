function [v, pi, acc_R] = sarsa(model, maxit, maxeps, alpha, epsilon)
rand('seed',287)
% initialize the value function
Q = zeros(model.stateCount, 4);
v = zeros(model.stateCount, 1);
v_ = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
acc_R = zeros(maxeps,1);

for i = 1:maxeps,
    % every time we reset the episode, start at the given startState
    s = model.startState;
    prob1 = rand;
    if prob1 < epsilon  % epsilon greedily
        a = datasample([1 2 3 4], 1); % radom action
    else
        [~,a] = max(Q(s,:));  % optimal action
    end
    
    for j = 1:maxit,
        % PICK AN ACTION
        p = 0;
        r = rand;
        for s_ = 1:model.stateCount,
            p = p + model.P(s, s_, a);
            if r <= p,
                break;
            end
        end
        
        % pick next action epsilon greedily
        prob2 = rand;
        if prob2 < epsilon
            a_ = datasample([1 2 3 4], 1);
        else
            [~,a_] = max(Q(s_,:));
        end  
        acc_R(i) = acc_R(i) + model.R(s,a); % accumulated reward
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a)+model.gamma*Q(s_,a_)-Q(s,a));
        [v_(s),pi(s)] = max(Q(s,:));
        s = s_; a = a_;
        
        % once reach goal state, start a new episode
        if s == model.goalState
            fprintf('%d: reached the goal state! Start Again!\n', i)
            break;
        end
    end
    if max(abs(v_-v))<0.001
       fprintf('value function converged after %d episodes!', i)
       break;
    end
    v = v_;
end

