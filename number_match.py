import re

def number_match_pair(pair): return number_match(pair[0], pair[1])

def number_match(source, target): return max(number_match_s(source, target, False), number_match_s(source, target, True))

def number_match_s(source, target, seps):
    
    def get_numbers(s, seps):
        return set([  
                    re.sub( '[0|,|\.|:|/]' if seps else '0', '',n) 
                    for n in  
                    re.findall('[\d|,|\.|:|/]+' if seps else '[\d]+', s)
                  ]).difference({''})
        
    nums = {'source': get_numbers(source, seps), 'target':get_numbers(target, seps)}
    
    total_nums = len(nums['source']) + len(nums['target'])
    num_symdiff = len(nums['source'].symmetric_difference(nums['target']))
    num_union = len(nums['source'].union(nums['target']))
    num_intersect = len(nums['source'].intersection(nums['target']))
    
    assert num_symdiff <= total_nums
    
    # score 0 by default
    if   total_nums == 0                      : score = 0.0
    
    #give a positive score if all numbers agree
    elif total_nums != 0 and num_symdiff == 0 : score = round(1.0-(1.0+num_union)**(-0.3333),2)
    
    #give a small penalty if there is a single number on one side 
    elif total_nums == 1 and num_symdiff == 1 : score = -0.2
    
    # give a large penalty for big mismatches, but reward if there are also a lot of matches
    # bounded between -1 and +1
    elif total_nums >  1 and num_symdiff >=  1: score = - ( float(num_symdiff) - float(num_intersect))/float(num_union) 
        
    assert score >= -1 and score <=1    
    
    s = (score + 1.0)*0.5
    
    assert s >= 0.0 and s <=1.0 

    return 6*s**5-15*s**4+10*s**3
