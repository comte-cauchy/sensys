%SCCs_Humans	Returns the orientation of the SCCs in humans, relative to Reid's plane
% Provides the orientation of the canals, according to the results of Blanks et al.
%	(R. H. Blanks, I. S. Curthoys, and C. H. Markham. Planar relationships of
%	 the semicircular canals in man. Acta Otolaryngol  80:185-196, 1975.)
% The orientation of the vectors indicates the direction of stimulation of
% the corresponding canal.
%
%	Call: Canals = SCCs_Humans();
%
%	ThH, 13-Jan-05 
%	Ver 1.0
%*****************************************************************
function CanalInfo = SCCs_Humans()

Canals(1).side = 'right';
Canals(1).rows = {'horizontal canal'; 'anterior canal'; 'posterior canal'};
Canals(1).orientation = [	.365	 .158	-.905
    .652	 .753	-.017
    .757	-.561	 .320];
Canals(2).side = 'left';
Canals(2).rows = {'hor'; 'ant'; 'post'};
Canals(2).orientation = [-.365	  .158 	 .905
    -.652	  .753 	 .017
    -.757 	 -.561	-.320];

% Normalize the canal-vectors (only a tiny correction):
for i = 1:2
    for j = 1:3
        Canals(i).orientation(j,:) = ...
            Canals(i).orientation(j,:) / norm( Canals(i).orientation(j,:) );
    end
end

CanalInfo = Canals;
