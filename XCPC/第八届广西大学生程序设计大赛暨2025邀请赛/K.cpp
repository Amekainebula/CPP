#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
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
struct in
{
    ld l, r;
    bool is() { return l <= r; }
};

in get(ld ax, ld ay, ld bx, ld by, ld cx, ld cy, ld r)
{
    ld dx = bx - ax;
    ld dy = by - ay;
    ld fx = ax - cx;
    ld fy = ay - cy;

    ld A = dx * dx + dy * dy;
    ld B = 2 * (fx * dx + fy * dy);
    ld C = fx * fx + fy * fy - r * r;
    ld D = B * B - 4 * A * C; // B^2-4AC
    if (D < 0)
        return {1e9, -1e9};//无交点

    ld sqrtD = sqrt(D);
    ld t1 = (-B - sqrtD) / (2 * A);
    ld t2 = (-B + sqrtD) / (2 * A);
    if (t1 > t2)
        swap(t1, t2);
    return {max((ld)0.0, t1), min((ld)1.0, t2)};//返回左右端点,为线段总长的百分比
}

ld f(ld ax, ld ay, ld bx, ld by)
{
    return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

void Murasame()
{
    ld x1, y1, r1, x2, y2, r2;
    cin >> x1 >> y1 >> r1 >> x2 >> y2 >> r2;
    ld a, b, c, d;
    cin >> a >> b >> c >> d;

    ld len = f(a, b, c, d);
    vc<in> segs;

    in iv1 = get(a, b, c, d, x1, y1, r1);
    in iv2 = get(a, b, c, d, x2, y2, r2);

    if (iv1.is())
        segs.pb(iv1);

    if (iv2.is())
        segs.pb(iv2);

    auto cmp = [](in &a, in &b)
    { return a.l < b.l; };
    sort(all(segs), cmp);
    vc<in> pre;
    for (auto &i : segs)
    {
        if (pre.empty() || i.l > pre.back().r)//为空或分开的两段
        {
            pre.pb(i);
        }
        else//其余的为有重合，合并
        {
            pre.back().r = max(pre.back().r, i.r);
        }
    }
    
    ld ans = 0;
    for (auto &i : pre)
    {
        ans += (i.r - i.l) * len;
    }

    cout << fixed << setprecision(10) << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}