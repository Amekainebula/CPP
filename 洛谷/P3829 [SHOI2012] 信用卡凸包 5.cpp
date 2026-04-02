#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x,y,z) for(int(x)=(y);(x)<=(z);++(x))
#define ffg(x,y,z) for(int(x)=(y);(x)>=(z);--(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
//#define endl endl << flush 
#define endl '\n' 
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
class Point
{
public:
    double x, y;
    Point() : x(0), y(0) {}
    Point(double x, double y) : x(x), y(y) {}
};

class TuBao
{
public:
    vector<Point> p;
    TuBao(vector<Point> _p) : p(_p) { init(); }
    double cross(Point a1, Point a2, Point b1, Point b2)
    {
        return (a2.x - a1.x) * (b2.y - b1.y) - (a2.y - a1.y) * (b2.x - b1.x);
    }
    double dis(Point a, Point b) { return hypot(a.x - b.x, a.y - b.y); }
    bool cmp(Point a, Point b)
    {
        double temp = cross(p[1], a, p[1], b);
        if (temp > 1e-9)
            return true;
        if (fabs(temp) < 1e-9 && dis(p[1], a) < dis(p[1], b))
            return true;
        return false;
    }

    void init()
    {
        int n = p.size() - 1;
        for (int i = 2; i <= n; i++)
        {
            if (p[i].y < p[1].y || (p[i].y == p[1].y && p[i].x < p[1].x))
            {
                swap(p[1], p[i]);
            }
        }
        sort(p.begin() + 2, p.begin() + n + 1, [this](Point a, Point b)
             { return this->cmp(a, b); });
    }

    double lenth()
    {
        int n = p.size() - 1;
        vector<Point> q(n + 2);
        q[1] = p[1];
        int cnt = 1;
        for (int i = 2; i <= n; i++)
        {
            while (cnt > 1 && cross(q[cnt - 1], q[cnt], q[cnt], p[i]) <= 0)
                cnt--;
            cnt++;
            q[cnt] = p[i];
        }
        q[cnt + 1] = p[1]; // 闭合
        double ans = 0;
        for (int i = 1; i <= cnt; i++)
            ans += dis(q[i], q[i + 1]);
        return ans;
    }

    double area()
    {
        int n = p.size() - 1;
        vector<Point> q(n + 2);
        q[1] = p[1];
        int cnt = 1;
        for (int i = 2; i <= n; i++)
        {
            while (cnt > 1 && cross(q[cnt - 1], q[cnt], q[cnt], p[i]) <= 0)
                cnt--;
            cnt++;
            q[cnt] = p[i];
        }
        q[cnt + 1] = p[1];
        double A = 0;
        for (int i = 1; i <= cnt; i++)
        {
            A += q[i].x * q[i + 1].y - q[i + 1].x * q[i].y;
        }
        return fabs(A) / 2.0;
    }
};
void Murasame()
{
    
    
    
    
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}