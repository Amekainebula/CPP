// #include <bits/stdc++.h>
// #define int long long
// #define ull unsigned long long
// #define i128 __int128
// #define ld long double
// #define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
// #define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
// #define pb push_back
// #define eb emplace_back
// #define pii pair<int, int>
// #define vc vector
// #define vi vector<int>
// #define fi first
// #define se second
// #define all(x) x.begin(), x.end()
// #define all1(x) x.begin() + 1, x.end()
// #define INF 0x7fffffffffffffff
// #define inf 0x7fffffff
// #define endl '\n'
// #define WA AC
// #define TLE AC
// #define MLE AC
// #define RE AC
// #define CE AC
// using namespace std;
// const string AC = "Accepted";
// ld n, t;
// ld sv = 0, st = 0;
// bool cmp(pair<ld, ld> a, pair<ld, ld> b) { return a.se < b.se; }
// void Murasame()
// {
//     cin >> n >> t;
//     vc<pair<ld, ld>> v(n + 1);
//     ld maxx = -1e9, minn = 1e9;
//     ff(i, 1, n) cin >> v[i].fi, sv += v[i].fi;
//     ff(i, 1, n)
//     {
//         cin >> v[i].se;
//         st += v[i].se * v[i].fi;
//         maxx = max(maxx, v[i].se);
//         minn = min(minn, v[i].se);
//     }
//     sort(all1(v), cmp);
//     if (n == 1)
//     {
//         cout << (st == t ? 0 : sv) << endl;
//         return;
//     }
//     if (maxx < t || minn > t)
//     {
//         cout << 0 << endl;
//         return;
//     }
//     auto check = [&](ld mid, int id)
//     {
//         ld temp = (st - mid * v[id].se) / (sv - mid);
//         return temp <= t;
//     };
//     if (st / sv != t)
//     {
//         if (st / sv > t)
//         {
//             ffg(i, n, 1)
//             {
//                 if ((st - v[i].fi * v[i].se) / (sv - v[i].fi) - t > 1e-7)
//                 {
//                     st -= v[i].fi * v[i].se;
//                     sv -= v[i].fi;
//                     continue;
//                 }
//                 ld l = 0, r = v[i].fi;
//                 while (r - l > 1e-9)
//                 {
//                     ld mid = (l + r) / 2;
//                     if (check(mid, i))
//                         r = mid;
//                     else
//                         l = mid + 1e-9;
//                     // cout << mid << endl;
//                     //  if( (st - mid * v[i].se) / (sv - mid)==t)
//                 }
//                 sv -= l;
//                 cout << sv << endl;
//                 return;
//             }
//         }
//         else
//         {
//             ff(i, 1, n)
//             {
//                 if ((st - v[i].fi * v[i].se) / (sv - v[i].fi) - t < -1e-7)
//                 {
//                     st -= v[i].fi * v[i].se;
//                     sv -= v[i].fi;
//                     continue;
//                 }
//                 ld l = 0, r = v[i].fi;
//                 while (r - l > 1e-7)
//                 {
//                     ld mid = (l + r) / 2;
//                     if (check(mid, i))
//                         l = mid + 1e-7;
//                     else
//                         r = mid;
//                 }
//                 sv -= l;
//                 cout << sv << endl;
//                 return;
//             }
//         }
//     }
//     cout << sv << endl;
// }
// signed main()
// {
//     ios::sync_with_stdio(false);
//     cin.tie(0);
//     cout.tie(0);
//     int _T = 1;
//     // cin >> _T;
//     while (_T--)
//     {
//         Murasame();
//     }
//     return 0;
// }
#include <bits/stdc++.h>
#define fi first
#define endl '\n'
#define il inline
#define se second
#define pb push_back
#define INF 0x3f3f3f3f
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
#ifdef ONLINE_JUDGE
#define debug(...) 0
#else
#define debug(...) fprintf(stderr, __VA_ARGS__), fflush(stderr)
#endif
const int N = 3e6 + 10;
struct Node
{
    double a, b;
} a[N];
int n;
double T, ans, s;
signed main()
{
    ios::sync_with_stdio(0);
    cin.tie(0), cout.tie(0);
    cout << fixed << setprecision(6);
    cin >> n >> T;
    for (int i = 1; i <= n; i++)
        cin >> a[i].a;
    for (int i = 1; i <= n; i++)
        cin >> a[i].b, a[i].b -= T, ans += a[i].a * a[i].b;
    if (ans < 0)
        for (int i = 1; i <= n; i++)
            a[i].b *= -1;
    sort(a + 1, a + n + 1, [&](Node x, Node y)
         { return x.b < y.b; });
    ans = 0;
    for (int i = 1; i <= n; i++)
        if (s + a[i].a * a[i].b <= 0)
            s += a[i].a * a[i].b, ans += a[i].a;
        else
        {
            ans += -s / a[i].b;
            break;
        }
    cout << fabs(ans) << endl;
    return 0;
}