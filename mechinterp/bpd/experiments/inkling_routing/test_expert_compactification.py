import itertools
import random
import unittest

from expert_compactification import route, certify_box, fixed_derivative, path_terms


class ExpertCompactificationTests(unittest.TestCase):
    def test_box_corners(self):
        rng = random.Random(17)
        checked = 0
        for _ in range(300):
            z = [rng.uniform(-2,2) for _ in range(6)]
            b = [rng.uniform(-.2,.2) for _ in z]
            writes = [[rng.uniform(-.1,.1) for _ in z] for _ in range(3)]
            ranges = [(-.5,.7)]*3
            cert = certify_box(z,b,writes,ranges,k=2)
            if cert['certified']:
                for g in itertools.product(*ranges):
                    changed = [x+sum(t*w[e] for t,w in zip(g,writes)) for e,x in enumerate(z)]
                    self.assertEqual(set(route(changed,b,2)[0]),set(cert['selected']))
                    checked += 1
        self.assertGreater(checked,500)
        print('certified box corners:',checked)

    def test_missing_and_ties(self):
        self.assertFalse(certify_box([2,1,0],[0]*3,[],[],k=2,allowed=[1,2])['certified'])
        self.assertFalse(certify_box([2,1,1],[0]*3,[],[],k=2)['certified'])
        self.assertTrue(certify_box([2,1,1],[0]*3,[],[],k=2,allowed=[0,1])['certified'])
        with self.assertRaises(ValueError):
            fixed_derivative([2,1,1],[0]*3,[[1],[2],[3]],[1],[1,2,3],k=2)

    def test_derivative(self):
        rng = random.Random(42)
        for _ in range(400):
            z = [rng.uniform(-2,2) for _ in range(6)]
            b = [rng.uniform(-.2,.2) for _ in z]
            v = [[rng.uniform(-1,1) for _ in range(3)] for _ in z]
            dv = [[rng.uniform(-1,1) for _ in range(3)] for _ in z]
            w = [rng.uniform(-1,1) for _ in z]
            a = [.3,-.2,.5]
            d = fixed_derivative(z,b,v,a,w,k=3,content_dot=dv)
            h = min(1e-5,d['gap']/10)
            def value(t):
                _,p = route([x+t*y for x,y in zip(z,w)],b,3)
                return 8*sum(p[e]*sum(a[j]*(v[e][j]+t*dv[e][j]) for j in range(3)) for e in range(6))
            self.assertAlmostEqual(d['total'],(value(h)-value(-h))/(2*h),places=6)
        print('finite derivative comparisons: 400')

    def test_factorial(self):
        rng = random.Random(8)
        for _ in range(300):
            z = [rng.random() for _ in range(6)]
            b = [rng.random() for _ in z]
            _,p = route(z,b,2)
            _,q = route(z,b,2,allowed=[0,2,4])
            v = [[rng.random(),rng.random()] for _ in z]
            u = [[rng.random(),rng.random()] for _ in z]
            r = path_terms(p,q,v,u)
            for j in range(2):
                self.assertAlmostEqual(r['total'][j],sum(r[x][j] for x in ('route_only','content_only','interaction')))
        print('factorial closure comparisons: 300')

    def test_zero_effect_and_invalid(self):
        d = fixed_derivative([2,1,0],[0]*3,[[4],[4],[4]],[1],[2,3,4],k=2)
        self.assertAlmostEqual(d['total'],0)
        with self.assertRaises(ValueError):
            route([1,2],[0,0],2,[1,1])
        with self.assertRaises(ValueError):
            certify_box([1,2],[0,0],[[1,1]],[(1,0)],1)
        with self.assertRaises(ValueError):
            path_terms([.2,.2],[.5,.5],[[1],[2]],[[1],[2]])


if __name__ == '__main__':
    unittest.main()
