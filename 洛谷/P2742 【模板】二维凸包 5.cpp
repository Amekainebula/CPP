#include <bits/stdc++.h>
using namespace std;
using i64 = long long;

constexpr long double eps = 1e-6;

struct Point
{
    long double x, y;
    Point(long double a = 0, long double b = 0) : x(a), y(b) {}
    bool operator<(const Point &p) { return x != p.x ? x < p.x : y < p.y; }
    Point operator+(const Point &p) { return Point(x + p.x, y + p.y); }
    Point operator-(const Point &p) { return Point(x - p.x, y - p.y); }
    long double len() { return sqrtl(x * x + y * y); }
    long double operator*(const Point &p) { return x * p.x + y * p.y; }
    long double Angle(Point &p) { return acos((*this) * p / (len() * p.len())); }
    long double OuterProduct(Point p) { return x * p.y - p.x * y; }
    bool same(Point &p) { return x == p.x && y == p.y; }
    long long dis(Point &p) { return ((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y)); }
    long double dis2(Point &p) { return sqrtl((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y)); }
    long double area(Point &p1, Point &p2) { return abs((p1 - (*this)).OuterProduct(p2 - p1)) / 2; }
    friend istream &operator>>(istream &in, Point &p)
    {
        in >> p.x >> p.y;
        return in;
    }
    friend ostream &operator<<(ostream &out, const Point &p)
    {
        out << p.x << " " << p.y;
        return out;
    }
};

class ConvexHull
{
public:
    vector<Point> p;
    ConvexHull(vector<Point> &_p)
    {
        int n = _p.size();
        sort(_p.begin(), _p.end(), [&](const Point &p1, const Point &p2)
             { return p1.x != p2.x ? p1.x < p2.x : p1.y < p2.y; });
        vector<int> vis(n), stk;
        stk.push_back(0);
        p.push_back(_p[0]);
        for (int i = 1; i < n; i++)
        {
            while (stk.size() >= 2 && (_p[stk.back()] - _p[stk[stk.size() - 2]])
                                              .OuterProduct(_p[i] - _p[stk.back()]) <= eps)
            {
                vis[stk.back()] = 0;
                stk.pop_back();
                p.pop_back();
            }
            vis[i] = 1;
            stk.push_back(i);
            p.push_back(_p[i]);
        }

        int low = stk.size();
        for (int i = n - 2; i >= 0; i--)
        {
            if (!vis[i])
            {
                while (stk.size() > low && (_p[stk.back()] - _p[stk[stk.size() - 2]])
                                                   .OuterProduct(_p[i] - _p[stk.back()]) <= eps)
                {
                    vis[stk.back()] = 0;
                    stk.pop_back();
                    p.pop_back();
                }
                vis[i] = 1;
                stk.push_back(i);
                p.push_back(_p[i]);
            }
        }
    }
    long long diameter()
    {
        if (p.size() <= 3)
        {
            return p[0].dis(p[1]);
        }
        else
        {
            long long ans = 0;
            int j = 2;
            for (int i = 0; i < p.size() - 1; i++)
            {
                while (p[j].area(p[i], p[i + 1]) <= p[j % (p.size() - 1) + 1].area(p[i], p[i + 1]))
                {
                    j = j % (p.size() - 1) + 1;
                }

                ans = max(ans, (i64)max(p[i].dis(p[j]), p[i + 1].dis(p[j])));
            }

            return ans;
        }
    }

    long double length()
    {
        double ans = 0;
        for (int i = 0; i < p.size(); ++i)
        {
            // cout << p[i] << endl;
            ans += p[i].dis2(p[(i + 1) % p.size()]);
        }
        return ans;
    }

    long double area()
    {
        long double a = 0;
        int m = p.size();
        for (int i = 0; i < m; ++i)
        {
            a += p[i].x * p[(i + 1) % m].y - p[(i + 1) % m].x * p[i].y;
        }
        return abs(a) / 2.0;
    }
};

void solve()
{
    int n;
    cin >> n;
    vector<Point> a(n);
    for (int i = 0; i < n; ++i)
    {
        cin >> a[i];
    }
    ConvexHull ch(a);
    cout << ch.diameter() << endl;
    cout << fixed << setprecision(2) << ch.length() << endl;
    cout << fixed << setprecision(2) << ch.area() << endl;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t = 1;
    while (t--)
        solve();
    return 0;
}
