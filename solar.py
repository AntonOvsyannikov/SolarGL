from EngineGL import *
from collections import namedtuple

Planet = namedtuple('Planet',
                    'name, tex, real_s, draw_s, real_orb, inclination, longitude, start_rot '
                    'slope, ars_period, self_period'
                    )

planets = [Planet(*p) for p in [
    ['Mercury', '2k_mercury.jpg', 0.38, 0.6, 58, 7, 48, 0, 0, 87, 25],
    ['Venus', '2k_venus_surface.jpg', 0.95, 0.9, 108, 3.4, 76, 0, 177, 224, 58],
    ['Earth', '2k_earth_daymap.jpg', 1.0, 1.0, 150, 0, 174, 0, 23, 365, 1],
    ['Mars', '2k_mars.jpg', 0.53, 0.7, 228, 1.8, 49, 0, 25, 686, 1],
    ['Jpiter', '2k_jupiter.jpg', 11.2, 1.9, 778, 1.3, 100, 0, 3, 4330, 0.4],
    ['Saturn', '2k_saturn.jpg', 9.45, 1.7, 1427, 2.5, 113, 30, 26, 10746, 0.4],
    ['Uranus', '2k_uranus.jpg', 4.0, 1.3, 2886, 0.8, 74, 0, 97, 30588, 0.7],
    ['Neptune', '2k_neptune.jpg', 3.88, 1.2, 4498, 1.8, 131, 0, 28, 59799, 0.6],
]]

# planets_to_display = [planets[i] for i in (1, 2, 5)]
planets_to_display = planets

Ring = namedtuple('Ring', 'tex, sz_int, sz_ext')

rings = {n: Ring(*r) for n, r in {
    'Saturn': ['rings.jpg', 1.15, 2.41]
}.iteritems()}

time = 0.0


def EventHandler(scene, event):
    global time
    if isinstance(event, MouseEvent): return Scene.DO_YOURSELF
    if isinstance(event, SceneEvent) and event.event == CAMERA_MOVE:
        scene.Camera.LookAt()

    if isinstance(event, AnimateEvent):
        time += event.time_passed

        intn_period = 3.0
        intn = cos(2 * pi * time / intn_period)
        scene.Sun_light.intensity = 1.0 + 0.2 * intn
        scene.Sun_mat.emission = 0.85 + 0.15 * intn
        scene.Rings_mat.emission = scene.Sun_mat.emission
        scene.Sun_atm_mat.emission = scene.Sun_mat.emission

        scene.Sun.Rotate(0, 0, -360 * event.time_passed / 60.0)
        scene.Sun_atm.Rotate(0, 0, 360 * event.time_passed / 30.0)

        ars_period = 20.0
        self_period = 5.0

        for p in planets_to_display:
            sun_pivot = scene[p.name + '_sunpivot']
            planet_system = scene[p.name + '_system']
            planet = scene[p.name]

            rot_sun = 360 * 365.0 / p.ars_period * event.time_passed / ars_period
            rot_self = 360 / p.self_period * event.time_passed / self_period

            sun_pivot.Rotate(0, 0, rot_sun)
            planet_system.Rotate(-rot_sun, 0,0)

            planet.Rotate(0, 0, rot_self)



            # return "y:{0} pp:{1} r:{2}".format(*[round(c,1) for c in camera.rotation])


g_scene = Scene(
    SceneOptions(name='Solar System', size=101, texture_dir='textures/solar/',
                 rotation_mode=RM_EUL, ambient_intensity=0.05),

    EventHandler,

    Camera().Name('Camera').Move(0, 0, 10).LookAt(),

    Texture('2k_stars_milky_way.jpg').Name('Space_tex'),
    Material(emission=1).Name('Space_mat'),
    Sphere().Name('Space').Size(100).Rotate(0, -150, 0).Texture('Space_tex').Color(BLACK).Material('Space_mat').Flip(),

    Texture('2k_sun.jpg').Name('Sun_tex'),
    Material(emission=1).Name('Sun_mat'),
    Material(emission=1, transparent=True).Name('Sun_atm_mat'),
    Light(type=POINT).Name('Sun_light'),
    Sphere().Name('Sun').Material('Sun_mat').Texture('Sun_tex').Size(1.4).Rotate(0, 7, 0),
    Sphere().Name('Sun_atm').Material('Sun_atm_mat').Color(BLACK).Alpha(0.4).Texture('Sun_tex').Size(1.42).Rotate(0, 30, 0),

    Material(emission=1.0,double_sided=True, transparent=True).Name('Rings_mat'),

)


orb = 0.8
for p in planets_to_display:

    orb += 0.8 * p.draw_s
    draw_s = p.draw_s * 0.4

    g_scene.Add(
        Texture(p.tex).Name(p.name + '_tex'),
        Pivot().Name(p.name + '_sunpivot').Rotate(p.longitude, p.inclination, p.start_rot),
        Circle(r=orb, width=0, segments=96).Color((0.2, 0.2, 0.1, 1.0)).BindTo(p.name + '_sunpivot'),
        Pivot().Name(p.name + '_system').Move(orb, 0, 0).BindTo(p.name + '_sunpivot')
            .Size(draw_s).Rotate(0, p.slope, 0),
        Pivot().Name(p.name).BindTo(p.name + '_system'),
        Sphere().Texture(p.name + '_tex').BindTo(p.name),
        #Box((0, 0.5, 0), (0.05, 0.4, 0.05)).BindTo(pp.name),
    )

    if p.name in rings:
        r = rings[p.name]
        g_scene.Add(
            Texture(r.tex).Name(p.name + '_rings_tex'),
            Circle(r=0.5 * r.sz_int, width=0.5 * (r.sz_ext - r.sz_int))
                .Texture(p.name + '_rings_tex').Color(BLACK).Alpha(0.8).Material('Rings_mat').BindTo(p.name + '_system'),
        )


# ShowSceneInWindow(g_scene)
print "Rendering..."
RenderSceneToFile('solar.png', g_scene)
print "Done!"