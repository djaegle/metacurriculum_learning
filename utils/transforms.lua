require 'torch'

local transforms = {}

function transforms.spatialTransform(config)
   -- Returns the spatial transformation function used to augment the dataset
   -- on the fly. Allows either batch- or instance-level randomization of augmentation
   -- parameters.
   local transformFunction

   if config.augMode == 'Cir2010' then
      -- TODO: implement this augmentation strategy
      -- Epochwise augmentation, so initialize the random parameters outside of the function
      -- All parameters sampled uniformly over the specified range
      -- Elastic deformation params sigma=[5.0,6.0], alpha=[36.0,38.0] (see Simard et al 2003)
      -- Rotation/horiz shearing: beta=[-7.5 deg, 7.5 deg] for 1 and 7, beta=[-15 deg, 15 deg] for others
      -- Horiz/vert scaling: gamma = [15,20], given as [1-gamma/100,1+gamma/100], independent scaling in x and y
      -- transformFunction =
   elseif config.augMode == 'Cir2012' then
      -- TODO: implement this augmentation strategy
      -- More elaborate: changes bounding box on samples before transforming,
      -- and returns each different-sized one to a different subnetwork.
      -- Low implement priority.
   elseif config.augMode == 'Wan2013' then
      -- TODO: implement this augmentation strategy
      -- Random Cropping to 24x24
      -- Rotate/scale up to 15%
   else
      -- Identity transformation
      -- e.g. for Goodfellow et al Maxout Nets
      transformFunction = function(sample) return sample end
   end

   return transformFunction
end

function transforms.dataNormTransform(config)
   -- Returns the normalizing transformation used on data.
   local normFunction

   if config.normMode == 'mnistZeroMean' then
      normFunction = function(sample)
         return {
            input = sample.input/127.5 - 1.0,
            target = sample.target,
         }
      end
   else
      -- Identity transformation
      normFunction = function(sample) return sample end
   end

   return normFunction
end

return transforms
