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
const int N = 1e6 + 6;
void Murasame()
{
    int n, k;
    cin >> n >> k;
    vi a(n + 1), b(n + 1), c(n + 1);
    ff(i, 1, n) cin >> a[i];
    b = a;
    sort(all1(b));
    int t = b[k - 1];
    int cnt = 0;
    ff(i, 1, n)
    {
        if (a[i] <= t)
        {
            c[++cnt] = a[i];
        }
    }
    int has = cnt - k + 1;
    int l = 1, r = cnt;
    bool ok = 1;
    while (l < r)
    {
        if (c[l] != c[r])
        {
            if (has <= 0)
            {
                ok = 0;
                break;
            }
            if (c[l] == t)
            {
                l++;
                has--;
            }
            else if (c[r] == t)
            {
                r--;
                has--;
            }
            else
            {
                ok = 0;
                break;
            }
            continue;
        }
        l++;
        r--;
    }
    if (ok)
    {
        cout << "YES" << endl;
    }
    else
    {
        cout << "NO"<< endl;
    }
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