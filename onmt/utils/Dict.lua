local Dict = torch.class("Dict")

function Dict:__init(data)
  self.idxToLabel = {}
  self.labelToIdx = {}
  self.frequencies = {}
  self.freqTensor = nil

  -- Special entries will not be pruned.
  self.special = {}

  if data ~= nil then
    if type(data) == "string" then -- File to load.
      self:loadFile(data)
    else
      self:addSpecials(data)
    end
  end
end

--[[ Return the number of entries in the dictionary. ]]
function Dict:size()
  --return #self.idxToLabel
  -- alway return Realsize (not getn)
  return self:getRealsize()
end

--[[ Return the number of entries in the dictionary. ]]
function Dict:getRealsize()
  count = 0
  for idx, label in pairs(self.idxToLabel) do
	count = count + 1
  end
  return count
end

--[[ Load entries from a file. ]]
function Dict:loadFile(filename)
  local reader = onmt.utils.FileReader.new(filename)

  -- add specials to dict (it you have other specials, plze insert here
  self:addSpecials({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD, 
					onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})

  while true do
    local fields = reader:next()

    if not fields then
      break
    end

    local label = fields[1]
    local idx = tonumber(fields[2])

    self:add(label, idx)
  end

  reader:close()
end

--[[ get sorted idxToLabel by idx ]]
function Dict:getSortIdx()

  sort_idx = {}
  for idx,label in pairs(self.idxToLabel) do
          table.insert(sort_idx,{idx=idx, label=label})
  end
  table.sort(sort_idx,function(a,b) return a.idx < b.idx end)
  return sort_idx
end

--[[ Write entries to a file. ]]
function Dict:writeFile(filename)
  local file = assert(io.open(filename, 'w'))

  -- get sorted IdxToLabel and write it file (preserve idx and order)
  for _,line in ipairs(self:getSortIdx()) do
    i = line.idx
	label = line.label
    if self.frequencies then
      file:write(label .. ' ' .. i .. ' ' .. (self.frequencies[i] or 0) .. '\n')
    elseif self.freqTensor then
      file:write(label .. ' ' .. i .. ' ' .. self.freqTensor[i] .. '\n')
    else
      file:write(label .. ' ' .. i .. '\n')
    end
  end

  file:close()
end

--[[ Drop or serialize the frequency tensor. ]]
function Dict:prepFrequency(keep)
  if not keep then
    self.freqTensor = nil
  else
    self.freqTensor = torch.Tensor(self.frequencies)
  end
  self.frequencies = nil
end

--[[ Lookup `key` in the dictionary: it can be an index or a string. ]]
function Dict:lookup(key)
  if type(key) == "string" then
    return self.labelToIdx[key]
  else
    return self.idxToLabel[key]
  end
end

--[[ Mark this `label` and `idx` as special (i.e. will not be pruned). ]]
function Dict:addSpecial(label, idx, frequency)
  idx = self:add(label, idx, frequency)
  table.insert(self.special, idx)
end

--[[ Mark all labels in `labels` as specials (i.e. will not be pruned). ]]
function Dict:addSpecials(labels)
  for i = 1, #labels do
    self:addSpecial(labels[i], nil, 0)
  end
end

--[[ Check if idx is index for special label ]]
function Dict:isSpecialIdx(idx)
  for _,v in ipairs(self.special) do
    if idx == v then
      return true
    end
  end
  return false
end

--[[ Set the frequency of a vocab. ]]
function Dict:setFrequency(label, frequency)
  local idx = self.labelToIdx[label]
  if idx then
    self.frequencies[idx] = frequency
  end
end

--[[ Add `label` in the dictionary. Use `idx` as its index if given. ]]
function Dict:add(label, idx, frequency)
  if not frequency then
    frequency = 1
  end
  if idx ~= nil then
    self.idxToLabel[idx] = label
    self.labelToIdx[label] = idx
  else
    idx = self.labelToIdx[label]
    if idx == nil then
      idx = #self.idxToLabel + 1
      self.idxToLabel[idx] = label
      self.labelToIdx[label] = idx
    end
  end

  if self.frequencies[idx] == nil then
    self.frequencies[idx] = frequency
  else
    self.frequencies[idx] = self.frequencies[idx] + frequency
  end

  return idx
end

--[[ Return a new dictionary with the `size` most frequent entries. ]]
function Dict:prune(size, preserve_idx)
  if size >= self:getRealsize()-#self.special then
    return self
  end
  -- frequnecy change from sequence list to hash list
  -- so handling changed from Tensor to sort hash

  -- Only keep the `size` most frequent entries.
  -- for idx,label in pairs(self.idxToLabel) do
  --  _G.logger:info('idx: ' .. idx .. 'freq :' .. self.frequencies[idx])
  --end
  --local freq = torch.Tensor(self.frequencies)
  --local _, idx = torch.sort(freq, 1, true)

  local sum_old_freq = 0
  local sum_new_freq = 0
  sort_freq = {}
  for idx,freq in pairs(self.frequencies) do
          table.insert(sort_freq,{idx=idx, freq=freq})
          -- sum old freq
          sum_old_freq = sum_old_freq + freq
  end
  -- sort by freq
  table.sort(sort_freq,function(a,b) return a.freq > b.freq end)

  local newDict = Dict.new()

  -- Add special entries in all cases.
  --print('special label: ' .. #self.special )
  for i = 1, #self.special do
    local thevocab = self.idxToLabel[self.special[i]]
    local thefreq = self.frequencies[self.special[i]]
    newDict:addSpecial(thevocab, nil, thefreq)
    --print('special label: ' .. thevocab .. 'freq :' .. thefreq )
  end


  local count = 0
  for _,line in ipairs(sort_freq) do
    idx = line.idx
    freq = line.freq
    if count >= size then
      break	  
	end 

    if not self:isSpecialIdx(idx) then
      if preserve_idx then
        newDict:add(self.idxToLabel[idx], idx, self.frequencies[idx])
      else
        newDict:add(self.idxToLabel[idx], nil, self.frequencies[idx])
      end
      sum_new_freq = sum_new_freq + self.frequencies[idx]
      count = count + 1
      --_G.logger:info('count: ' .. count .. 'idx: ' .. idx .. 'freq :' .. self.frequencies[idx] .. 'label' .. self.idxToLabel[idx] .. 'size: ' .. size)
    end
  end

  --local i = 1
  --local count = 0
  --while count ~= size do
  --  if not self:isSpecialIdx(idx[i]) then
  --    if preserve_idx then
  --      newDict:add(self.idxToLabel[idx[i]], idx, self.frequencies[idx[i]])
  --    else
  --      newDict:add(self.idxToLabel[idx[i]], nil, self.frequencies[idx[i]])
  --    end
  --    count = count + 1
  --  end
  --  i = i + 1
  --end

  -- set UNK frequency
  
  newDict:setFrequency(onmt.Constants.UNK_WORD, sum_old_freq - sum_new_freq)
  --newDict:setFrequency(onmt.Constants.UNK_WORD, freq:sum()-torch.Tensor(newDict.frequencies):sum())

  return newDict
end

--[[ Return a new dictionary with entries appearing at least `minFrequency` times. ]]
-- TODO: remove Tensor handle(frequnecies is not sequence of list) (see prune function)
function Dict:pruneByMinFrequency(minFrequency, preserve_idx)
  if minFrequency < 2 then
    return self
  end

  local freq = torch.Tensor(self.frequencies)
  local sortedFreq, idx = torch.sort(freq, 1, true)

  local newDict = Dict.new()

  -- Add special entries in all cases.
  for i = 1, #self.special do
    local thevocab = self.idxToLabel[self.special[i]]
    local thefreq = self.frequencies[self.special[i]]
    newDict:addSpecial(thevocab, nil, thefreq)
  end

  for i = 1, self:size() do
    if sortedFreq[i] < minFrequency then
      break
    end
    if preserve_idx then
      newDict:add(self.idxToLabel[idx[i]], idx, sortedFreq[i])
    else
      newDict:add(self.idxToLabel[idx[i]], nil, sortedFreq[i])
    end
  end

  -- set UNK frequency
  newDict:setFrequency(onmt.Constants.UNK_WORD, freq:sum()-torch.Tensor(newDict.frequencies):sum())

  return newDict
end


--[[ Reset frequency to 0 (expect 1~4 PAD, UNK, BOS, EOS ]]
function Dict:resetFrequencies()

  --for i = 1, 4 do
  --  local token = self.idxToLabel[i]
  --  local idx = self.labelToIdx[token]
  --  _G.logger:info('idx: ' .. idx .. 'freq :' .. self.frequencies[idx] .. 'label' .. token)
  --end

  -- after specials (unk, eos, bos, pad)
  -- reset frequnecies to 0
  for i = 5, self:size() do
    local token = self.idxToLabel[i]
    local idx = self.labelToIdx[token]
    local frequency = 0
    if idx then
      self.frequencies[idx] = 0
    end
    --_G.logger:info('idx: ' .. idx .. 'freq :' .. self.frequencies[idx] .. 'label' .. token)
  end

end

--[[ Only Leave elements which frequencies is bigger 0 ]]
function Dict:compactZeroFrequencies()
  local newDict = Dict.new()

  -- Add special entries in all cases.
  --print('special label: ' .. #self.special )
  for i = 1, #self.special do
    local thevocab = self.idxToLabel[self.special[i]]
    local thefreq = self.frequencies[self.special[i]]
    newDict:addSpecial(thevocab, nil, thefreq)
    --print('special label: ' .. thevocab .. 'freq :' .. thefreq )
  end

  for i = #self.special + 1, self:size() do
    local token = self.idxToLabel[i]
    local idx = self.labelToIdx[token]
    if idx then
      frequency = self.frequencies[idx]
    else
      frequency = 0
    end
    if frequency > 0 then
      --_G.logger:info('compact result idx ' .. token .. 'token ' .. idx .. 'fre ' .. frequency)
      newDict:add(token, idx, frequency)
    end
    --_G.logger:info('compact' .. newDict:size())

  end

  return newDict
end

--[[ Add frequency to current dictionary from provided dictionary ]]
function Dict:getFrequencies(dict)
  local newDict = Dict.new()

  for i = 1, dict:size() do
    local token = dict.idxToLabel[i]
    local idx = self.labelToIdx[token]
    local frequency = 0
    if idx then
      frequency = self.frequencies[idx]
    end
    newDict:add(token, i)
    newDict.frequencies[i] = frequency
  end
  -- set UNK frequency
  newDict:setFrequency(onmt.Constants.UNK_WORD,
                      torch.Tensor(self.frequencies):sum()-torch.Tensor(newDict.frequencies):sum());
  return newDict
end

--[[
  Convert `labels` to indices. Use `unkWord` if not found.
  Optionally insert `bosWord` at the beginning and `eosWord` at the end.
]]
function Dict:convertToIdx(labels, unkWord, bosWord, eosWord)
  local vec = {}

  if bosWord ~= nil then
    table.insert(vec, self:lookup(bosWord))
  end

  for i = 1, #labels do
    local idx = self:lookup(labels[i])
    if idx == nil then
      idx = self:lookup(unkWord)
    end
    table.insert(vec, idx)
  end

  if eosWord ~= nil then
    table.insert(vec, self:lookup(eosWord))
  end

  return torch.IntTensor(vec)
end

--[[ Convert `idx` to labels. If index `stop` is reached, convert it and return. ]]
function Dict:convertToLabels(idx, stop)
  local labels = {}

  for i = 1, #idx do
    table.insert(labels, self:lookup(idx[i]))
    if idx[i] == stop then
      break
    end
  end

  return labels
end

return Dict
