import copy

# landmark_names = ['Bed', 'DiningTable', 'StoveBurner', 'Toilet', 'Sink', 'Desk','Drawer','Shelf',
#                         'CounterTop','Television','Sofa','SideTable','CoffeeTable','ShelvingUnit','ArmChair','TVStand']
# Word_Dict = {
#     'Bed': 'bed', 'DiningTable': 'dining table', 'StoveBurner': 'stove', 'Toilet': 'toilet', 'Sink': 'sink',
#     'Desk': 'Desk', 'CounterTop':'kitchen table', 'Sofa':'sofa', 'Television':'television','Drawer':'drawer',
#      'SideTable':'side table', 'CoffeeTable':'coffee table','ShelvingUnit':'shelving unit','ArmChair':'arm chair','TVStand':'tv stand',
#      'Shelf':'shelf'}

in_landmark_names = ['dining table','sofa','tv monitor']
out_landmark_names = ['desk','drawer','side table','coffee table','bed','arm chair']
landmark_names = in_landmark_names+out_landmark_names

def choose_ladmark(objects):
    landmarks = []
    temp_ln = copy.deepcopy(landmark_names)
    temp_ln.remove('Shelf')
    temp_ln.remove("Drawer")
    temp_ln.remove("TVStand")
    for obj in objects:
        if obj['objectType'] in temp_ln:
            cp = obj["position"]
            flag = True
            for l in landmarks:
                if abs(l['cp']['x']-cp['x'])+ abs(l['cp']['z']-cp['z']) < 0.7 and l['name']==obj['objectType']: 
                    l['ID'].append(obj['objectId'])
                    flag = False
                    break
            if flag:
                landmarks.append(dict(cp = cp, name=obj['objectType'],ID = [obj['objectId']]))
            
    for obj in objects:
        if obj['objectType'] == 'Shelf' or obj['objectType'] =='Drawer' or obj['objectType'] == 'TVStand':
            cp = obj["position"]
            flag = True
            for l in landmarks:
                if abs(l['cp']['x']-cp['x'])+ abs(l['cp']['z']-cp['z']) < 0.7:
                    if obj['objectType'] == 'TVStand':
                        if l['name'] == 'DiningTable': 
                            l['ID'].append(obj['objectId'])
                    else:
                        l['ID'].append(obj['objectId'])
                    flag = False
            if flag:
                landmarks.append(dict(cp = cp, name=obj['objectType'],ID = [obj['objectId']]))
    visible_landmark_name = []
    for l in landmarks:
        if l['name'] not in visible_landmark_name:
            visible_landmark_name.append(l['name'])
    return landmarks,visible_landmark_name

