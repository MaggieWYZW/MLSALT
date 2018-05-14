function [v, pi, acc_R] = qLearning(model, maxit, maxeps, alpha, epsilon)
rand('seed',287)
% initialize the value function
Q = zeros(model.stateCount, 4);
v = zeros(model.stateCount, 1);
v_ = zeros(model.stateCount, 1);
pi = ones(model.stateCount, 1);
acc_R = zeros(maxeps, 1);

for i = 1:maxeps,
    s = model.startState;

    for j = 1:maxit,
        % PICK AN ACTION
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
        acc_R(i) = acc_R(i) + model.R(s,a);
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a) + model.gamma*max(Q(s_,:)) - Q(s,a));
        [v_(s), pi(s)] = max(Q(s,:));
        s = s_;
        
        % start new episode after reaching goal state
        if s == model.goalState
            fprintf('%d: reached the goal state! Start Again!\n', i)
            break;
        end
    end
    
%     if max(abs(v_-v))<0.00000000001
%        fprintf('value function converged after %d episodes!', i)
%        break;
%     end
    v = v_;
end