% Entities
student(s1). student(s2).
assessor(a1). assessor(a2).
reviewer(r1). reviewer(r2).

% Education levels (higher number = higher education)
education(s1, 2). education(s2, 3).
education(a1, 4). education(a2, 3).
education(r1, 5). education(r2, 4).

% Expertise
expertise(a1, math). expertise(a2, physics).
expertise(r1, math). expertise(r2, physics).
expertise(s1, math). expertise(s2, physics).

% Matching constraints
% Assessor must have expertise in student's subject and higher education
valid_assessor(S, A) :- student(S), assessor(A), expertise(A, Subj), expertise(S, Subj), education(A, EA), education(S, ES), EA > ES.

% Reviewer must have expertise in student's subject and higher education
valid_reviewer(S, R) :- student(S), reviewer(R), expertise(R, Subj), expertise(S, Subj), education(R, ER), education(S, ES), ER > ES.

% Assign one assessor and one reviewer per student
{ match_assessor(S, A) : valid_assessor(S, A) } = 1 :- student(S).
{ match_reviewer(S, R) : valid_reviewer(S, R) } = 1 :- student(S).

% Output
#show match_assessor/2.
#show match_reviewer/2.