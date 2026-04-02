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
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
bool isprime[6000005];
int prime[6000005];
int cnt = 0;
int fsum[6000005], bsum[6000005];
void euler()
{
    isprime[0] = isprime[1] = 1;
    for (int i = 2; i <= 6e6; i++)
    {
        if (!isprime[i])
        {
            prime[++cnt] = i;
            fsum[cnt] = fsum[cnt - 1] + i;
        }
        for (int j = 1; j <= cnt && i * prime[j] <= 6e6; j++)
        {
            isprime[i * prime[j]] = 1;
            if (i % prime[j] == 0)
                break;
        }
    }
}
void Murasame()
{
    int n;
    cin >> n;
    ff(i, 1, n + 1) bsum[i] = 0;
    vi a(n + 1);
    ff(i, 1, n) cin >> a[i];
    sort(all1(a));
    ffg(i, n, 1) bsum[i] = bsum[i + 1] + a[i];
    auto check = [&](int mid)
    {
        return bsum[mid+1] - fsum[n - mid] >= 0;
    };
    int l = 0, r = n;
    while (l < r)
    {
        int mid = (l + r) >> 1;
        if (check(mid))
            r = mid;
        else
            l = mid + 1;
    }
    cout << l << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    euler();
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}