from ai2thor.controller import Controller

c = Controller()
agent_view = c.last_event.frame
event = c.step('ToggleMapView')
map_view = event.frame
c.step('ToggleMapView') 