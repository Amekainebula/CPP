#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define double long double
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
// #define endl endl << flush
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
const double pi = 6.283185307179586 / 2;
double dis(double x1, double y1, double x2, double y2)
{
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}
double get(double angle)
{
    while (angle < 0)
        angle += 2 * pi;
    while (angle > 2 * pi)
        angle -= 2 * pi;
    return angle;
}
vc<double> minsp(vc<pair<double, double>> &a, int x, int y)
{
    vc<double> angle;
    for (int i = 1; i < a.size(); i++)
    {
        double dx = a[i].fi - x, dy = a[i].se - y;
        double angle1 = atan2(dy, dx);
        angle.pb(get(angle1));
    }
    sort(all(angle));
    vc<double> diff;
    for (int i = 0; i < angle.size(); i++)
    {

        if (i == angle.size() - 1)
        {
            diff.pb(get(angle[0] + 2 * pi - angle[i]));
        }
        else
        {
            diff.pb(get(angle[i + 1] - angle[i]));
        }
    }
    return diff;
}
void Murasame()
{
    double n, x, y;
    cin >> n >> x >> y;
    vc<pair<double, double>> v(n + 1), w(1, {0, 0});
    double mx = -inf;
    ff(i, 1, n)
    {
        cin >> v[i].fi >> v[i].se;
        mx = max(mx, dis(v[i].fi, v[i].se, x, y));
    }
    ff(i, 1, n)
    {
        if (dis(v[i].fi, v[i].se, x, y) == mx)
        {
            w.pb({v[i].fi, v[i].se});
        }
    }
    double ans = 0.0;
    vc<double> diff = minsp(w, x, y);
    for (double x : diff)
    {
        ans = max(ans, x);
        // cout << fixed << setprecision(10) << x << endl;
    }
    cout << fixed << setprecision(10) << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}