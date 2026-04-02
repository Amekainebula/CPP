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
    int k, a, b, x, y;
    cin >> k >> a >> b >> x >> y;
    int ans = 0;
    int mn, now;
    if (x == y) //
    {
        now = k - min(a, b);
        if (now >= 0)
        {
            ans += now / x;
            now -= ans * x;
            if (now >= 0)
            {
                ans++;
            }
        }
    }
    else if (x < y)
    {
        if (a == b)
        {
            now = k - a;
            if (now >= 0)
            {
                ans += now / x;
                now -= ans * x;
                if (now >= 0)
                {
                    ans++;
                }
            }
        }
        else if (a < b)
        {
            now = k - a;
            if (now >= 0)
            {
                ans += now / x;
                now -= ans * x;
                if (now >= 0)
                {
                    ans++;
                }
            }
        }
        else if (a > b)
        {
            now = k - a;
            if (now >= 0)
            {
                ans += now / x;
                now -= ans * x;
                if (now >= 0)
                {
                    ans++;
                }
            }
            k -= ans * x;
            now = k - b;
            if (now >= 0)
            {
                ans += now / y + 1;
            }
        }
    }
    else // x > y
    {
        if (a == b)
        {
            now = k - a;
            if (now >= 0)
            {
                ans += now / y;
                now -= ans * y;
                if (now >= 0)
                {
                    ans++;
                }
            }
        }
        else if (a > b)
        {
            now = k - b;
            if (now >= 0)
            {
                ans += now / y;
                now -= ans * y;
                if (now >= 0)
                {
                    ans++;
                }
            }
        }
        else // a < b
        {
            now = k - b;
            if (now >= 0)
            {
                ans += now / y;
                now -= ans * y;
                if (now >= 0)
                {
                    ans++;
                }
            }
            k -= ans * y;
            now = k - a;
            if (now >= 0)
            {
                ans += now / x + 1;
            }
        }
    }
    cout << ans << endl;
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